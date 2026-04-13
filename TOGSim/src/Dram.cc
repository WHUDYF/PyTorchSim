#include "Dram.h"

#include <iostream>

namespace {

static bool is_power_of_2_u32(uint32_t n) { return n != 0 && (n & (n - 1)) == 0; }

static uint32_t floor_log2_u32(uint32_t n) {
  uint32_t r = 0;
  while (n >>= 1)
    ++r;
  return r;
}

/** Smallest power of two >= n (n >= 1). */
static uint32_t next_power_of_2_u32(uint32_t n) {
  if (n <= 1)
    return 1;
  --n;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

/** Bytes/s effective GB/s and avg-per-channel utilization % for a window of `window_cycles` DRAM ticks. */
struct DramBwSnapshot {
  double bandwidth_gbs = 0;
  double util_avg_ch_pct = 0;
};

DramBwSnapshot make_dram_bw_snapshot(long long total_rw_transactions, uint64_t window_cycles,
                                     uint32_t n_ch, uint32_t req_size, uint32_t n_bl,
                                     double dram_freq_mhz) {
  DramBwSnapshot out;
  if (window_cycles == 0 || n_ch == 0)
    return out;
  const double tx = static_cast<double>(total_rw_transactions);
  const double w = static_cast<double>(window_cycles);
  const double bytes_per_cycle = tx * static_cast<double>(req_size) / w;
  out.bandwidth_gbs = bytes_per_cycle * dram_freq_mhz / 1000.0;
  const double avg_per_ch = tx / static_cast<double>(n_ch);
  out.util_avg_ch_pct = avg_per_ch * 100.0 * static_cast<double>(n_bl) / (2.0 * w);
  return out;
}

}  // namespace

new_addr_type Dram::partition_dram_address(new_addr_type raw_addr) const {
  if (_req_size == 0 || _n_ch_per_partition == 0)
    return raw_addr;
  const new_addr_type tx = raw_addr >> _tx_log2;
  const new_addr_type q = tx / _n_ch_per_partition;
  return static_cast<new_addr_type>(q << _tx_log2);
}

uint32_t Dram::get_channel_id(mem_fetch* access) {
  uint32_t channel_in_partition = 0;
  if (_n_ch_per_partition > 1) {
    const new_addr_type tx = static_cast<new_addr_type>(access->get_addr() >> _tx_log2);
    new_addr_type rest_high;
    unsigned init_index = 0;
    if (is_power_of_2_u32(_n_ch_per_partition)) {
      const unsigned lb = floor_log2_u32(_n_ch_per_partition);
      rest_high = tx >> lb;
      init_index = static_cast<unsigned>(tx & (_n_ch_per_partition - 1u));
    } else {
      /* gpgpu-sim "gap" channels: quotient / remainder split at txn granularity. */
      rest_high = tx / _n_ch_per_partition;
      init_index = static_cast<unsigned>(tx % _n_ch_per_partition);
    }
    /* ipoly_hash_function only implements 16/32/64 (see Hashing.cc); fold like addrdec IPOLY + mod when needed. */
    const uint32_t poly_n = next_power_of_2_u32(std::max(16u, _n_ch_per_partition));
    const uint32_t poly_use = std::min(poly_n, 64u);
    channel_in_partition =
        static_cast<uint32_t>(ipoly_hash_function(rest_high, init_index, poly_use)) % _n_ch_per_partition;
  }

  const uint32_t channel_id =
      channel_in_partition + static_cast<uint32_t>(access->get_numa_id() % _n_partitions) * _n_ch_per_partition;
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
  _tx_log2 = static_cast<int>(std::log2(_req_size));

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

  if (_n_ch == 0)
    return;
  const int iv = _config.dram_print_interval;
  if (iv <= 0)
    return;
  const uint64_t cc = *_core_cycles;
  if (cc % static_cast<uint64_t>(iv) != 0 || cc == 0)
    return;

  const double f_mhz = static_cast<double>(_config.dram_freq_mhz);
  const uint64_t w = static_cast<uint64_t>(iv);
  long long r_all = 0;
  long long w_all = 0;
  for (int ch = 0; ch < _n_ch; ch++) {
    const long long r = _mem[ch]->interval_reads();
    const long long wtxn = _mem[ch]->interval_writes();
    r_all += r;
    w_all += wtxn;
    const DramBwSnapshot bw =
        make_dram_bw_snapshot(r + wtxn, w, 1u, _req_size, _n_bl, f_mhz);
    spdlog::trace(
        "[DRAM] ch {} | BW {:.2f} GB/s, {:.2f}% util | {} reads, {} writes (interval {} cycles)",
        ch, bw.bandwidth_gbs, bw.util_avg_ch_pct, r, wtxn, w);
  }
  const DramBwSnapshot bw_all =
      make_dram_bw_snapshot(r_all + w_all, w, _n_ch, _req_size, _n_bl, f_mhz);
  spdlog::info(
      "[DRAM] all {} ch | BW {:.2f} GB/s, {:.2f}% util (avg/ch) | {} reads, {} writes (interval {} cycles)",
      _n_ch, bw_all.bandwidth_gbs, bw_all.util_avg_ch_pct, r_all, w_all, w);
  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->reset_interval_bw_counters();
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
  const addr_type raw_addr = request->get_addr();
  const addr_type target_addr = partition_dram_address(raw_addr);
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
  spdlog::info("========= DRAM stat =========");
  if (_n_ch == 0)
    return;

  for (int ch = 0; ch < _n_ch; ch++) {
    _mem[ch]->finalize_once();
  }

  spdlog::trace("=== Ramulator2 stats (channels 0.. {}) ===", _n_ch - 1);
  for (int ch = 0; ch < _n_ch; ch++) {
    std::cout << "--- channel " << ch << " ---\n";
    _mem[ch]->print_stats_yaml(std::cout);
  }
  std::cout.flush();

  const uint64_t cycles = *_core_cycles;
  if (cycles == 0)
    return;
  const double f_mhz = static_cast<double>(_config.dram_freq_mhz);
  spdlog::info("[DRAM] per-channel avg BW ({} sim cycles):", cycles);
  long long tr_all = 0;
  long long tw_all = 0;
  for (int ch = 0; ch < _n_ch; ch++) {
    const long long tr = _mem[ch]->total_reads();
    const long long tw = _mem[ch]->total_writes();
    tr_all += tr;
    tw_all += tw;
    const DramBwSnapshot bw =
        make_dram_bw_snapshot(tr + tw, cycles, 1u, _req_size, _n_bl, f_mhz);
    spdlog::info(
        "[DRAM] ch {} | avg BW {:.2f} GB/s, {:.2f}% util | {} reads, {} writes",
        ch, bw.bandwidth_gbs, bw.util_avg_ch_pct, tr, tw);
  }
  const DramBwSnapshot bw_all = make_dram_bw_snapshot(
      tr_all + tw_all, cycles, _n_ch, _req_size, _n_bl, f_mhz);
  spdlog::info(
      "[DRAM] all ch 0..{} | avg BW {:.2f} GB/s, {:.2f}% util (avg/ch) | {} reads, {} writes",
      _n_ch - 1, bw_all.bandwidth_gbs, bw_all.util_avg_ch_pct, tr_all, tw_all);
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
