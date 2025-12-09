#include <fstream>
#include <chrono>
#include <filesystem>

#include "Simulator.h"
#include "TileGraphParser.h"
#include "helper/CommandLineParser.h"

namespace fs = std::filesystem;
namespace po = boost::program_options;

const char* env_value = std::getenv("TOGSIM_EAGER_MODE");
bool isDryRun = (env_value != nullptr && std::string(env_value) == "1");

void launchKernel(Simulator* simulator, std::string onnx_path, std::string attribute_path, std::string config_path, cycle_type request_time=0, int partiton_id=0) {
  auto graph_praser = TileGraphParser(onnx_path, attribute_path, config_path);
  std::unique_ptr<TileGraph>& tile_graph = graph_praser.get_tile_graph();
  tile_graph->set_arrival_time(request_time ? request_time : simulator->get_core_cycle());
  spdlog::info("[Scheduler {}] Register graph path: {} operation: {} at {}", partiton_id, onnx_path, tile_graph->get_name(), simulator->get_core_cycle());

  simulator->schedule_graph(partiton_id, std::move(tile_graph));
}

Simulator* create_simulator(std::string config_path) {
  YAML::Node config_yaml;
  if (!loadConfig(config_path, config_yaml))
    exit(1);
  SimulationConfig config = initialize_config(config_yaml);

  auto simulator = new Simulator(config);
  return simulator;
}

int until(Simulator *simulator, cycle_type until_cycle) {
  return simulator->until(until_cycle);
}

void interactive_mode(Simulator* simulator) {
  std::string command;

  std::cout << "[" << simulator->get_core_cycle() << "] TOGSim> ";
  while (std::getline(std::cin, command)) {

    std::istringstream iss(command);
    std::string token;
    // Parse the first part of the command (e.g., "launch", "until", "quit")
    iss >> token;
    if (token == "launch") {
      std::string onnx_path, attribute_path, config_path;
      cycle_type request_time = 0;
      int partition_id = 0;
      iss >> config_path >> onnx_path >> attribute_path >> request_time >> partition_id;

      // Check if both paths were provided
      if (onnx_path.empty() || attribute_path.empty()) {
        spdlog::error("Error: Please provide both ONNX path and Attribute path in the format: launch onnx/path attribute/path");
      } else {
        launchKernel(simulator, onnx_path, attribute_path, config_path, request_time, partition_id);
        std::cerr << "launch done" << std::endl;
      }
    } else if (token == "until") {
      cycle_type until_cycle;
      iss >> until_cycle;
      int reason;

      if (iss.fail()) {
        spdlog::error("Error: Please provide a valid cycle number after 'until'");
      } else {
        reason = simulator->until(until_cycle);
        std::cerr << " Until finished: " << reason << std::endl;
      }
    } else if (token == "cycle") {
      cycle_type current_cycle = simulator->get_core_cycle();
      std::cerr << "Current cycle: " << current_cycle << std::endl;
    }else if (token == "quit") {
      std::cerr << "Quit" << std::endl;
      break;
    } else {
      spdlog::error("Error: unknown command {} Available commands are: launch, until, quit.", token);
    }
    if (isDryRun)
      std::cout << "[" << simulator->get_core_cycle() << "] TOGSim> ";
  }
  simulator->cycle();
  if (simulator->get_core_cycle()==0)
    simulator->until(0);
  simulator->print_core_stat();
}

int main(int argc, char** argv) {
  auto start = std::chrono::high_resolution_clock::now();
  // parse command line argumnet
  CommandLineParser cmd_parser = CommandLineParser();
  cmd_parser.add_command_line_option<std::string>(
      "config", "Path for hardware configuration file");
  cmd_parser.add_command_line_option<std::string>(
      "models_list", "Path for the models list file");
  cmd_parser.add_command_line_option<std::string>(
      "attributes_list", "Path for the models list file");
  cmd_parser.add_command_line_option<std::string>(
      "log_level", "Set for log level [trace, debug, info], default = info");
  cmd_parser.add_command_line_option<std::string>(
      "mode", "choose \"trace\" moode and \"iteractive\" mode");
  try {
    cmd_parser.parse(argc, argv);
  } catch (const CommandLineParser::ParsingError& e) {
    spdlog::error(
        "Command line argument parrsing error captured. Error message: {}",
        e.what());
    throw(e);
  }

  std::string level = "info";
  cmd_parser.set_if_defined("log_level", &level);
  if (level == "trace")
    spdlog::set_level(spdlog::level::trace);
  else if (level == "debug")
    spdlog::set_level(spdlog::level::debug);
  else if (level == "info")
    spdlog::set_level(spdlog::level::info);

  std::string config_path;
  std::string onnx_path;
  std::string attribute_path;
  std::string execution_mode = "trace";

  /* Create simulator */
  cmd_parser.set_if_defined("config", &config_path);
  cmd_parser.set_if_defined("mode", &execution_mode);
  auto simulator = create_simulator(config_path);

  if (execution_mode.compare("trace") == 0) {
    /* Get needed info for launch kernel */
    cmd_parser.set_if_defined("models_list", &onnx_path);
    cmd_parser.set_if_defined("attributes_list", &attribute_path);

    /* launch kernels */
    launchKernel(simulator, onnx_path, attribute_path, config_path);
    simulator->run_simulator();
    if (simulator->get_core_cycle()==0)
      simulator->until(1);
    simulator->print_core_stat();
  } else if (execution_mode.compare("interactive") == 0) {
    /* Get onnx_path, attribute from user input, request_time */
    interactive_mode(simulator);
  }
  delete simulator;

  /* Simulation time measurement */
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  spdlog::info("Wall-clock time for simulation: {:2f} seconds", duration.count());
  return 0;
}
