#include "Core.h"

Core::Core(uint32_t id, SimulationConfig config)
    : _id(id),
      _config(config),
      _core_cycle(0),
      _stat_dma_cycle(0),
      _num_systolic_array_per_core(config.num_systolic_array_per_core),
      _dma(id, config.dram_req_size) {
  _sa_compute_pipeline.resize(_num_systolic_array_per_core);
  _stat_tot_sa_compute_cycle.resize(_num_systolic_array_per_core);
  _stat_sa_compute_cycle.resize(_num_systolic_array_per_core);
  _stat_tot_sa_compute_idle_cycle.resize(_num_systolic_array_per_core);
  _stat_sa_compute_idle_cycle.resize(_num_systolic_array_per_core);
  _stat_inst_count.resize(static_cast<size_t>(Opcode::COUNT), 0);
  _stat_tot_skipped_inst.resize(static_cast<size_t>(Opcode::COUNT), 0);
}

bool Core::can_issue(const std::shared_ptr<Tile>& op) {
  /* Check SRAM is enough to run tile */
  return _tiles.size() < 4  && !op->is_stonne_tile();
}

void Core::issue(std::shared_ptr<Tile> op) {
  if (op->get_instructions().size()){
    spdlog::trace("[{}][Core {}][TILE_SCHEDULED]",
      _core_cycle, _id);
  }
  for (const auto& inst : op->get_instructions()) {
    if (inst->is_ready())
      op->enqueue_ready(inst);
  }
  _tiles.push_back(std::move(op));
}

std::shared_ptr<Tile> Core::pop_finished_tile() {
  std::shared_ptr<Tile> result = std::make_unique<Tile>(Tile(Tile::Status::EMPTY));
  if (_finished_tiles.size() > 0) {
    result = std::move(_finished_tiles.front());
    _finished_tiles.pop();
  }
  return result;
}

std::queue<std::shared_ptr<Instruction>>& Core::get_compute_pipeline(int compute_type) {
  if (compute_type == VECTOR_UNIT)
    return _vu_compute_pipeline;
  else if (compute_type == MATMUL || compute_type == PRELOAD) {
    uint32_t sa_idx = _systolic_array_rr;
    _systolic_array_rr = (_systolic_array_rr + 1) % _num_systolic_array_per_core;
    return _sa_compute_pipeline.at(sa_idx);
  }
  else {
    spdlog::error("Undefined compute type");
    exit(EXIT_FAILURE);
  }
}

void Core::vu_cycle() {
  bool retry = true;
  while (retry) {
    if (!_vu_compute_pipeline.empty()) {
      _stat_vu_compute_cycle++;
      if(_vu_compute_pipeline.front()->finish_cycle <= _core_cycle) {
        cycle_type bubble = _vu_compute_pipeline.front()->bubble_cycle;
        _stat_vu_compute_idle_cycle += bubble;
        _stat_vu_compute_cycle = (bubble < _stat_vu_compute_cycle) ? (_stat_vu_compute_cycle - bubble) : 0;
        finish_instruction(_vu_compute_pipeline.front());
        _vu_compute_pipeline.pop();
      } else {
        retry = false;
      }
    } else {
      _stat_vu_compute_idle_cycle++;
      retry = false;
    }
  }
}

void Core::sa_cycle() {
  for (int i=0; i<_num_systolic_array_per_core; i++) {
    bool retry = true;
    while (retry) {
      if (!_sa_compute_pipeline.at(i).empty()) {
        if(_sa_compute_pipeline.at(i).front()->finish_cycle <= _core_cycle) {
          cycle_type bubble = _sa_compute_pipeline.at(i).front()->bubble_cycle;
          _stat_sa_compute_idle_cycle.at(i) += bubble;
          cycle_type& stat = _stat_sa_compute_cycle.at(i);
          stat = (bubble < stat) ? (stat - bubble) : 0;
          finish_instruction(_sa_compute_pipeline.at(i).front());
          _sa_compute_pipeline.at(i).pop();
        } else {
          _stat_sa_compute_cycle.at(i)++;
          retry = false;
        }
      } else {
        _stat_sa_compute_idle_cycle.at(i)++;
        retry = false;
      }
    }
  }
}

void Core::compute_cycle() {
  vu_cycle();
  sa_cycle();
}

void Core::dma_cycle() {
  /* Check finished dma operation */
  while(_dma_finished_queue.size()) {
    std::shared_ptr<Instruction>& instruction = _dma_finished_queue.at(0);
    assert(instruction->get_waiting_request()==0);

    /* Finish DMA read instruction */
    if (instruction->is_dma_read() && !instruction->is_async_dma())
      finish_instruction(instruction);

    /* Set tag table of async dma load */
    if (instruction->is_dma_read() && instruction->is_async_dma()) {
      auto& key = instruction->get_tag_id();
      assert(!_dma.get_tag_finish(instruction->subgraph_id, key));
      _dma.set_tag_finish(instruction->subgraph_id, key);
      spdlog::trace("[{}][Core {}] {} ASYNC FINISHED, subgraph_id: {} addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}",
                    _core_cycle, _id, opcode_to_string(instruction->get_opcode()),
                    instruction->subgraph_id, instruction->get_addr_name(),
                    fmt::format("[{}]", fmt::join(instruction->get_tag_id(), ", ")),
                    fmt::format("[{}]", fmt::join(instruction->get_tag_idx_list(), ", ")),
                    fmt::format("[{}]", fmt::join(instruction->get_tag_stride_list(), ", ")));
      for (auto & wait_inst : _dma.get_tag_waiter(instruction->subgraph_id, key)) {
        _dma.mark_tag_used(instruction->subgraph_id, key);
        finish_instruction(wait_inst);
      }
    }
    _dma_finished_queue.erase(_dma_finished_queue.begin());
  }

  if (_dma.is_finished()) {
    /* Finish instruction when it is DMA store */
    if (_dma.get_current_inst() != nullptr) {
      std::shared_ptr<Instruction> finished_inst = std::move(_dma.get_current_inst());
      if (finished_inst->is_dma_write()) {
        /* Only DMA write operation is finished! */
        finish_instruction(finished_inst);
      } else if (finished_inst->is_dma_read() && finished_inst->is_async_dma()) {
        /* Register tag table for async dma load */
        _dma.register_tag(finished_inst->subgraph_id, finished_inst->get_tag_id());
        finish_instruction(finished_inst);
      } else if(!finished_inst->is_dma_read()) {
        spdlog::error("[{}][Core {}] DMA instruction in not valid", _core_cycle, _id);
        exit(EXIT_FAILURE);
      } else if (finished_inst->get_opcode() == Opcode::BAR) {
        spdlog::trace("[{}][Core {}] {} FINISHED, addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _core_cycle, _id,
                      opcode_to_string(finished_inst->get_opcode()), finished_inst->get_addr_name(),
                      fmt::format("[{}]", fmt::join(finished_inst->get_tag_id(), ", ")),
                      fmt::format("[{}]", fmt::join(finished_inst->get_tag_idx_list(), ", ")),
                      fmt::format("[{}]", fmt::join(finished_inst->get_tag_stride_list(), ", ")));
      }
      /*Pass to waiting queue */
      _dma_waiting_queue[finished_inst.get()] = std::move(finished_inst);
    }

    /* Issue new DMA operation */
    if (!_ld_inst_queue.empty()) {
      std::shared_ptr<Instruction> inst = _ld_inst_queue.front();
      _dma.issue_tile(inst);
      _ld_inst_queue.pop();
    } else if (!_st_inst_queue.empty()) {
      std::shared_ptr<Instruction> inst = _st_inst_queue.front();
      _dma.issue_tile(inst);
      _st_inst_queue.pop();
    } else {
      /* DMA is idle */
      _stat_dma_idle_cycle++;
      return;
    }
  }
  /* Generate memfetch */
  auto access_vec = _dma.get_memory_access(_core_cycle, _config.icnt_injection_ports_per_core);
  for (auto access : *access_vec) {
    access->set_start_cycle(_core_cycle);
    _request_queue.push(access);
  }

  /* Increase dma stat cycle */
  _stat_dma_cycle++;
}

void Core::cycle() {
  /* Run compute unit and DMA unit */
  compute_cycle();
  dma_cycle();

  /* Increase core cycle counter */
  _core_cycle++;

  /* Iterate tile while an instruction is issued */
  bool issued = false;

  for (int i=0; i<_tiles.size() && !issued; i++) {
    auto& instructions = _tiles[i]->get_ready_instructions();
    for (auto it=instructions.begin(); it!=instructions.end();) {
      auto& inst = *it;
      /* Skip instruction is not ready  */
      //if (!inst->is_ready())
      //  continue;

      switch (inst->get_opcode()) {
        case Opcode::MOVIN:
          {
            /* Check another MOVIN with same tag is issued */
            auto& key = inst->get_tag_id();
            if (inst->is_sparse_inst()) {
              _dma.register_tag(inst->subgraph_id, key);
              _dma.set_tag_sparse(inst->subgraph_id, key);
              finish_instruction(inst);
              issued = true;
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              break;
            } else if (inst->is_async_dma() && _dma.tag_key_exist(inst->subgraph_id, key)) {
              bool finished = _dma.get_tag_finish(inst->subgraph_id, key);
              if (finished)
                finish_instruction(inst);
              else
                _dma.register_tag_waiter(inst->subgraph_id, key, inst);
              spdlog::trace("[{}][Core {}][SIKIPPED] {}, addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _core_cycle, _id,
                            opcode_to_string(inst->get_opcode()),
                            inst->get_addr_name(),
                            fmt::format("[{}]", fmt::join(inst->get_tag_id(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
              issued = true;
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              break;
            } else {
              spdlog::trace("[{}][Core {}][INST_ISSUED] {}, addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _core_cycle, _id,
                            opcode_to_string(inst->get_opcode()),
                            inst->get_addr_name(),
                            fmt::format("[{}]", fmt::join(inst->get_tag_id(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
              _ld_inst_queue.push(inst);
              issued = true;
              break;
            }
          }
        case Opcode::MOVOUT:
          spdlog::trace("[{}][Core {}][INST_ISSUED] {}, addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _core_cycle, _id,
                        opcode_to_string(inst->get_opcode()),
                        inst->get_addr_name(),
                        fmt::format("[{}]", fmt::join(inst->get_tag_id(), ", ")),
                        fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
                        fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
          _st_inst_queue.push(inst);
          issued = true;
          break;
        case Opcode::COMP:
          {
            auto& target_pipeline = get_compute_pipeline(inst->get_compute_type());
            if (target_pipeline.empty()) {
              inst->finish_cycle = _core_cycle + inst->get_compute_cycle();
              inst->bubble_cycle = inst->get_overlapping_cycle();
            } else {
              int overlapped_cycle = std::min(target_pipeline.back()->finish_cycle - _core_cycle, inst->get_overlapping_cycle());
              int bubble_cycle = inst->get_overlapping_cycle() - overlapped_cycle;
              inst->finish_cycle = target_pipeline.back()->finish_cycle + inst->get_compute_cycle() - overlapped_cycle;
              inst->bubble_cycle = bubble_cycle;
            }

            if (inst->get_compute_cycle() == 0) {
              inst->finish_instruction();
              static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
              _stat_tot_skipped_inst.at(static_cast<size_t>(inst->get_opcode()))++;
              instructions.erase(it);
            } else {
              spdlog::trace("[{}][Core {}][INST_ISSUED][SA {}] {}-{}, finsh at {}", _core_cycle, _id, _systolic_array_rr,
                            opcode_to_string(inst->get_opcode()), inst->get_compute_type(), inst->finish_cycle);
              target_pipeline.push(inst);
              issued = true;
              if (inst->get_compute_type()) {
                _stat_gemm_inst++;
              }
            }
          }
          break;
        case Opcode::BAR:
          {
            auto& key = inst->get_tag_id();
            uint32_t finished = _dma.get_tag_finish(inst->subgraph_id, key);
            if (finished == -1) {
              for (auto child_inst : inst->get_child_inst()) {
                if (child_inst->get_opcode() == Opcode::COMP && child_inst->get_compute_type() == MATMUL) {
                  child_inst->set_compute_cycle(0);
                }
              }
              finish_instruction(inst);
            } else if (finished != 0) {
              _dma.mark_tag_used(inst->subgraph_id, key);
              finish_instruction(inst);
            } else {
              _dma.register_tag_waiter(inst->subgraph_id, key, inst);
            }
            spdlog::trace("[{}][Core {}][INST_ISSUED] {},  addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}", _core_cycle, _id,
                            opcode_to_string(inst->get_opcode()), inst->get_addr_name(),
                            fmt::format("[{}]", fmt::join(inst->get_tag_id(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
                            fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
            issued = true;
          }
          break;
        default:
          spdlog::error("Undefined instruction opcode type");
          exit(EXIT_FAILURE);
      }

      if (issued) {
        _stat_inst_count.at(static_cast<size_t>(inst->get_opcode()))++;
        instructions.erase(it);
        break;
      }
      it++;
    }
  }

  /* Remove finshed tiles */
  bool retry = true;
  while (retry) {
    for (int i=0; i<_tiles.size() && !issued; i++) {
      if (_tiles[i]->all_insts_finshed()) {
        _tiles[i]->set_status(Tile::Status::FINISH);
        _finished_tiles.push(std::move(_tiles[i]));
        _tiles.erase(_tiles.begin() + i); // FIXME. Inefficient data structure
        /* Let's retry */
        break;
      }
    }
    retry = false;
  }
  if(_config.core_print_interval && _core_cycle % _config.core_print_interval == 0) {
    print_current_stats();
  }
}

void Core::finish_instruction(std::shared_ptr<Instruction>& inst) {
  if (inst->finished) {
    spdlog::error("[{}][Core {}][ERROR] {} inst already finished!!", _core_cycle, _id,
                  opcode_to_string(inst->get_opcode()));
    exit(EXIT_FAILURE);
  }
  inst->finish_instruction();
  static_cast<Tile*>(inst->get_owner())->inc_finished_inst();
  if (inst->get_opcode() == Opcode::COMP) {
    spdlog::trace("[{}][Core {}][INST_FINISHED] {}-{}",
      _core_cycle, _id, opcode_to_string(inst->get_opcode()), inst->get_compute_type());
  } else if (inst->get_opcode() != Opcode::BAR && inst->is_async_dma()){
    spdlog::trace("[{}][Core {}][ASYNC] {} subgraph_id: {} addr_name: {} tag_id: {} tag_idx_list: {} tag_stride_list: {}",
      _core_cycle, _id, opcode_to_string(inst->get_opcode()), inst->subgraph_id, inst->get_addr_name(),
      inst->get_tag_id(),
      fmt::format("[{}]", fmt::join(inst->get_tag_idx_list(), ", ")),
      fmt::format("[{}]", fmt::join(inst->get_tag_stride_list(), ", ")));
  } else if ((inst->get_opcode() == Opcode::MOVIN || inst->get_opcode() == Opcode::MOVOUT) && !inst->is_async_dma()) {
    spdlog::trace("[{}][Core {}][INST_FINISHED] {} addr_name: {}", _core_cycle, _id,
      opcode_to_string(inst->get_opcode()), inst->get_addr_name());
  }
}

bool Core::running() {
  bool running = false;
  running = running || _tiles.size() > 0;
  running = running || !_vu_compute_pipeline.empty();
  for (int i=0; i<_num_systolic_array_per_core;i++)
    running = running || !_sa_compute_pipeline.at(i).empty();
  running = running || !_dma_waiting_queue.empty() || !_dma_finished_queue.empty();
  running = running || !_dma.empty();
  running = running || !_ld_inst_queue.empty();
  running = running || !_st_inst_queue.empty();
  return running;
}

bool Core::has_memory_request() {
  return !_request_queue.empty();
}

void Core::pop_memory_request() {
  _request_queue.pop();
}

void Core::push_memory_response(mem_fetch* response) {
  Instruction* owner_inst = static_cast<Instruction*>(response->get_custom_data());
  assert(owner_inst->get_waiting_request());

  owner_inst->dec_waiting_request();
  if (!owner_inst->get_waiting_request()) {
    auto it = _dma_waiting_queue.find(owner_inst);
    if (it != _dma_waiting_queue.end()) {
      std::shared_ptr<Instruction> moved_inst = std::move(it->second);
      _dma_finished_queue.push_back(std::move(moved_inst));
      _dma_waiting_queue.erase(it);
    } else {
      assert(true || "Can't happend...!");
    }
  }
  _stat_mem_response++;
  delete response;
}

bool Core::can_issue_compute(std::shared_ptr<Instruction>& inst) {
  return inst->is_ready();
}

void Core::print_stats() {
  std::vector<float> sa_utilization;
  update_stats();
  spdlog::info("===== Instructions count =====");
  for (int i = 0; i < static_cast<size_t>(Opcode::COUNT); i++) {
    auto opcode  = static_cast<Opcode>(i);
    auto inst = _stat_inst_count.at(i);
    auto skipped = _stat_tot_skipped_inst.at(i);
    auto name = opcode_to_string(opcode);

    if (opcode == Opcode::COMP) {
      auto gemm   = _stat_gemm_inst;
      auto vector = inst - gemm;
      if (skipped)
        spdlog::info("Core [{}] : {:8} inst_count {} (GEMM: {}, Vector: {}), skipped inst_count {}",
            _id, name, inst, gemm, vector, skipped);
      else
        spdlog::info("Core [{}] : {:8} inst_count {} (GEMM: {}, Vector: {})",
            _id, name, inst, gemm, vector);
    }
    else {
      if (skipped)
        spdlog::info("Core [{}] : {:8} inst_count {}, skipped inst_count {}",
            _id, name, inst, skipped);
      else
        spdlog::info("Core [{}] : {:8} inst_count {}",
            _id, name, inst);
    }
  }
  spdlog::info("========= Core stat =========");
  for (int i=0; i<_num_systolic_array_per_core; i++)
    sa_utilization.push_back(static_cast<float>(_stat_tot_sa_compute_cycle.at(i) * 100) / _core_cycle);
  for (int i=0; i<_num_systolic_array_per_core; i++)
    spdlog::info("Core [{}] : Systolic array [{}] utilization(%) {:.2f}, active_cycles {}, idle_cycles {}", _id, i, sa_utilization.at(i),
      _stat_tot_sa_compute_cycle.at(i), _stat_tot_sa_compute_idle_cycle.at(i));
  float dram_bw = _config.dram_req_size * _stat_tot_mem_response * _config.core_freq_mhz / (_core_cycle * 1000); // B/cycle
  spdlog::info("Core [{}] : DMA active_cycles, {} DMA idle_cycles {}, DRAM BW {:.3f} GB/s ({} responses)", _id, _stat_tot_dma_cycle, _stat_tot_dma_idle_cycle, dram_bw, _stat_tot_mem_response);
  spdlog::info("Core [{}] : Vector unit utilization(%) {:.2f}, active cycle {}, idle_cycle {}", _id,
    static_cast<float>(_stat_tot_vu_compute_cycle * 100) / _core_cycle, _stat_tot_vu_compute_cycle, _stat_tot_vu_compute_idle_cycle);
  spdlog::info("Core [{}] : NUMA local memory: {} requests, remote memory: {} requests", _id, _stat_numa_local_access, _stat_numa_remote_access);
  spdlog::info("Core [{}] : Total_cycles {}", _id, _core_cycle);
}

void Core::print_current_stats() {
  std::vector<float> sa_utilization;
  for (int i=0; i<_num_systolic_array_per_core; i++)
    sa_utilization.push_back(static_cast<float>(_stat_sa_compute_cycle.at(i) * 100) / _config.core_print_interval);
  float dram_bw = _config.dram_req_size * _stat_mem_response * _config.core_freq_mhz / (_config.core_print_interval * 1000); // B/cycle
  auto level = spdlog::level::info;
  if(_id != 0)
    level = spdlog::level::debug;

  spdlog::info("========= Core stat =========");
  for (int i=0; i<_num_systolic_array_per_core; i++)
    spdlog::info("Core [{}] : Systolic array [{}] utilization(%) {:.2f}, active_cycles {}, idle_cycles {}", _id, i, sa_utilization.at(i),
      _stat_sa_compute_cycle.at(i), _stat_sa_compute_idle_cycle.at(i));
  spdlog::info("Core [{}] : DMA active_cycles {}, DMA idle_cycles {}, DRAM BW {:.3f} GB/s ({} responses)", _id, _stat_dma_cycle, _stat_dma_idle_cycle, dram_bw, _stat_mem_response);
  spdlog::info("Core [{}] : Vector unit Utilization(%) {:.2f}, active_cycles {}, idle_cycles {}", _id,
    static_cast<float>(_stat_vu_compute_cycle * 100) / _config.core_print_interval, _stat_vu_compute_cycle, _stat_vu_compute_idle_cycle);
  spdlog::info("Core [{}] : Total_cycles {}", _id, _core_cycle);
  update_stats();
}

void Core::update_stats() {
  for (int i=0; i<_num_systolic_array_per_core; i++) {
    _stat_tot_sa_compute_cycle.at(i) += _stat_sa_compute_cycle.at(i);
    _stat_tot_sa_compute_idle_cycle.at(i) += _stat_sa_compute_idle_cycle.at(i);
    _stat_sa_compute_cycle.at(i) = 0;
    _stat_sa_compute_idle_cycle.at(i) = 0;
  }

  _stat_tot_vu_compute_cycle += _stat_vu_compute_cycle;
  _stat_tot_dma_cycle += _stat_dma_cycle;
  _stat_tot_dma_idle_cycle += _stat_dma_idle_cycle;
  _stat_tot_mem_response += +_stat_mem_response;

  _stat_vu_compute_cycle = 0;
  _stat_dma_cycle = 0;
  _stat_dma_idle_cycle = 0;
  _stat_vu_compute_idle_cycle = 0;
  _stat_mem_response = 0;
}