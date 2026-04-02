#include "Dram.h"

uint32_t Dram::get_channel_id(mem_fetch* access) {
  uint32_t channel_id;
  if (_n_ch_per_partition >= 16)
    channel_id = ipoly_hash_function((new_addr_type)access->get_addr()/_req_size, 0, _n_ch_per_partition);
  else
    channel_id = ipoly_hash_function((new_addr_type)access->get_addr()/_req_size, 0, 16) % _n_ch_per_partition;

  channel_id += ((access->get_numa_id() % _n_partitions)* _n_ch_per_partition);
  return channel_id;
}

Dram::Dram(SimulationConfig config, cycle_type* core_cycle) {
  _core_cycles = core_cycle;
  _n_ch = config.dram_channels;
  _n_bl = config.dram_nbl;
  _req_size = config.dram_req_size;
  _n_partitions = config.dram_num_partitions;
  _n_ch_per_partition = config.dram_channels_per_partitions;
  _config = config;

  spdlog::info("[Config/DRAM] DRAM Bandwidth {} GB/s, Freq: {} MHz, Channels: {}, Request_size: {}B", config.max_dram_bandwidth(), config.dram_freq_mhz, _n_ch, _req_size);
  /* Initialize DRAM Channels */
  for (int ch = 0; ch < _n_ch; ch++) {
    m_to_crossbar_queue.push_back(std::queue<mem_fetch*>());
    m_from_crossbar_queue.push_back(std::queue<mem_fetch*>());
  }

  /* Initialize L2 cache */
  _m_caches.resize(_n_ch);
  if (config.l2d_type == L2CacheType::NOCACHE) {
    std::string name = "No cache";
    spdlog::info("[Config/L2Cache] No L2 cache");
    for (int ch = 0; ch < _n_ch; ch++)
      _m_caches[ch] = new NoL2Cache(name, _m_cache_config, ch, _core_cycles, &m_to_crossbar_queue[ch], &m_from_crossbar_queue[ch]);
  } else if (config.l2d_type == L2CacheType::DATACACHE) {
    std::string name = "L2 cache";
    _m_cache_config.init(config.l2d_config_str);
    spdlog::info("[Config/L2Cache] Total Size: {} KB, Partition Size: {} KB, Set: {}, Assoc: {}, Line Size: {}B Sector Size: {}B",
            _m_cache_config.get_total_size_in_kb() * _n_ch, _m_cache_config.get_total_size_in_kb(),
            _m_cache_config.get_num_sets(), _m_cache_config.get_num_assoc(),
            _m_cache_config.get_line_size(), _m_cache_config.get_sector_size());
    for (int ch = 0; ch < _n_ch; ch++)
      _m_caches[ch] = new L2DataCache(name, _m_cache_config, ch, _core_cycles, _config.l2d_hit_latency, _config.num_cores, &m_to_crossbar_queue[ch], &m_from_crossbar_queue[ch]);
  } else {
    spdlog::error("[Config/L2D] Invalid L2 cache type...!");
    exit(EXIT_FAILURE);
  }
}

DramRamulator2::DramRamulator2(SimulationConfig config, cycle_type* core_cycle) : Dram(config, core_cycle) {
  /* Initialize DRAM Channels */
  _mem.resize(_n_ch);
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch] = std::make_unique<Ramulator2>(
      ch, _n_ch, config.dram_config_path, "Ramulator2", _config.dram_print_interval, _n_bl,
      _req_size, config.dram_freq_mhz);
  }
  _tx_log2 = log2(_req_size);
  _tx_ch_log2 = log2(_n_ch_per_partition) + _tx_log2;
}

bool DramRamulator2::running() {
  for (int ch = 0; ch < _n_ch; ch++) {
    if (mem_fetch* req = _mem[ch]->return_queue_top())
      return true;
    if (mem_fetch* req = _m_caches[ch]->top())
      return true;
  }
  return false;
}

void DramRamulator2::cycle() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->cycle();

    // From Cache to DRAM
    if (mem_fetch* req = _m_caches[ch]->top()) {
      _mem[ch]->push(req);
      _m_caches[ch]->pop();
    }

    // From DRAM to Cache
    if (mem_fetch* req = _mem[ch]->return_queue_top()) {
      if(_m_caches[ch]->push(req))
        _mem[ch]->return_queue_pop();
    }
  }
}

void DramRamulator2::cache_cycle()  {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->cycle();
  }
}

bool DramRamulator2::is_full(uint32_t cid, mem_fetch* request) {
  return false; //m_from_crossbar_queue[cid].full(); Infinite length
}

void DramRamulator2::push(uint32_t cid, mem_fetch* request) {
  addr_type target_addr = (request->get_addr() >> _tx_ch_log2) << _tx_log2;
  request->set_addr(target_addr);
  m_from_crossbar_queue[cid].push(request);
}

bool DramRamulator2::is_empty(uint32_t cid) {
  return m_to_crossbar_queue[cid].empty();
}

mem_fetch* DramRamulator2::top(uint32_t cid) {
  assert(!is_empty(cid));
  return m_to_crossbar_queue[cid].front();
}

void DramRamulator2::pop(uint32_t cid) {
  assert(!is_empty(cid));
  m_to_crossbar_queue[cid].pop();
}

void DramRamulator2::print_stat() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->print(stdout);
  }
}

void DramRamulator2::print_cache_stats() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->print_stats();
  }
}

SimpleDRAM::SimpleDRAM(SimulationConfig config, cycle_type* core_cycle) : Dram(config, core_cycle) {
  /* Initialize DRAM Channels */
  spdlog::info("[SimpleDRAM] DRAM latecny: {}", config.dram_latency);
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem.push_back(std::make_unique<DelayQueue<mem_fetch*>>("SimpleDRAM", true, -1));
  }
  _latency =  config.dram_latency;
  _tx_log2 = log2(_req_size);
  _tx_ch_log2 = log2(_n_ch_per_partition) + _tx_log2;
}

bool SimpleDRAM::running() {
  for (int ch = 0; ch < _n_ch; ch++) {
    if (!_mem[ch]->queue_empty())
      return true;
    if (mem_fetch* req = _m_caches[ch]->top())
      return true;
  }
  return false;
}

void SimpleDRAM::cycle() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->cycle();

    // From Cache to DRAM
    if (mem_fetch* req = _m_caches[ch]->top()) {
      //spdlog::info("[Cache->DRAM] mem_fetch: addr={:#x}", req->get_addr());

      _mem[ch]->push(req, _latency);
      _m_caches[ch]->pop();
    }

    // From DRAM to Cache
    if (_mem[ch]->arrived()) {
      mem_fetch* req = _mem[ch]->top();
      req->set_reply();
      //spdlog::info("[DRAM->Cache] mem_fetch: addr={:#x}", req->get_addr());
      if(_m_caches[ch]->push(req))
        _mem[ch]->pop();
    }
  }
}

void SimpleDRAM::cache_cycle()  {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->cycle();
  }
}

bool SimpleDRAM::is_full(uint32_t cid, mem_fetch* request) {
  return false; //m_from_crossbar_queue[cid].full(); Infinite length
}

void SimpleDRAM::push(uint32_t cid, mem_fetch* request) {
  m_from_crossbar_queue[cid].push(request);
}

bool SimpleDRAM::is_empty(uint32_t cid) {
  return m_to_crossbar_queue[cid].empty();
}

mem_fetch* SimpleDRAM::top(uint32_t cid) {
  assert(!is_empty(cid));
  return m_to_crossbar_queue[cid].front();
}

void SimpleDRAM::pop(uint32_t cid) {
  assert(!is_empty(cid));
  m_to_crossbar_queue[cid].pop();
}

void SimpleDRAM::print_stat() {}

void SimpleDRAM::print_cache_stats() {
  for (int ch = 0; ch < _n_ch; ch++) {
    _m_caches[ch]->print_stats();
  }
}
