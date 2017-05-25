#!/usr/bin/env python
import argparse
from collections import defaultdict
from functools import partial
from itertools import repeat
import helpers as helpers
import json
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import os
from pprint import pprint
import shutil
import sys
from collections import defaultdict


def LoadGoogleBenchmarkJsonReportResults(filename):
  with open(filename) as data_file:
    data = json.load(data_file)
  return data


def PlotPerformanceOverParameter(parameter_values, performance_values):
  fig = plt.figure()
  plt.plot([1, 2, 3, 4])
  plt.ylabel('Bla.')
  return fig


def GeneratePerformancePlotOverParameters(benchmark_context, benchmark_data, real_cpu_freq_mhz):
  if benchmark_context["cpu_scaling_enabled"]:
    print "WARNING: Dynamic frequency scaling was active during the benchmarking!"
  if benchmark_context["library_build_type"] != "release":
    print "WARNING: Benchmarking was run on a non release build."

  cpu_freq_to_use_mhz = benchmark_context["mhz_per_cpu"]
  if real_cpu_freq_mhz > 0:
    cpu_freq_to_use_mhz = real_cpu_freq_mhz

  print("Use {} MHz as CPU frequency.".format(cpu_freq_to_use_mhz))

  figures_dict = dict()
  figures_runtime_dict = dict()
  figures_flops_dict = dict()
  figures_cycles_dict = dict()
  figures_memory_dict = dict()

  # Plot stuff.
  rcParams['font.family'] = 'sans-serif'
  rcParams['font.sans-serif'] = ['Gill Sans MT']

  colors = {'Baseline': '#284910', 'Fast': '#C51929', 'Other': '#ECC351'}
  bar_indent = {'Baseline': 0.1, 'Fast': 0.55, 'Other': 1.25}
  bar_width = 0.35

  y_label_dict = {"flops": "flops", "preallocated_memory_B":
                  "Allocated memory [MB]", "flops_p_cycle": "Performance [flops/cycle]", "cpu_time": "Runtime [s]", "cycles": "cycles", "preallocated_blocks": "Allocated blocks"}

  for case_results in benchmark_data:
    case_name = str.split(str(case_results[0]["name"]), '/')[1]

    split_case_name = str.split(case_name, '_')
    if len(split_case_name) > 2:
      exit("Your case name is wrong. It should have 2 parts separated by _")

    if split_case_name[0] not in figures_dict:
      print 'Creating figure for ' + split_case_name[0]
      fig, ax = plt.subplots()
      figures_dict[split_case_name[0]] = fig
    else:
      fig = figures_dict[split_case_name[0]]

    if split_case_name[0] not in figures_runtime_dict:
      print 'Creating runtime figure for ' + split_case_name[0]
      figr, axr = plt.subplots()
      figures_runtime_dict[split_case_name[0]] = figr
    else:
      figr = figures_runtime_dict[split_case_name[0]]

    if split_case_name[0] not in figures_memory_dict:
      print 'Creating memory plot for ' + split_case_name[0]
      figm, axm = plt.subplots()
      figures_memory_dict[split_case_name[0]] = figm
    else:
      figm = figures_memory_dict[split_case_name[0]]

    if split_case_name[0] not in figures_flops_dict:
      print 'Creating flop plot for ' + split_case_name[0]
      figf, axf = plt.subplots()
      figures_flops_dict[split_case_name[0]] = figf
    else:
      figf = figures_flops_dict[split_case_name[0]]

    if split_case_name[0] not in figures_cycles_dict:
      print 'Creating cycle plot for ' + split_case_name[0]
      figc, axc = plt.subplots()
      figures_cycles_dict[split_case_name[0]] = figc
    else:
      figc = figures_cycles_dict[split_case_name[0]]

    ax = fig.gca()
    axr = figr.gca()
    axm = figm.gca()
    axf = figf.gca()
    axc = figc.gca()

    print 'Benchmarking case: ' + split_case_name[0] + ', version: ' + split_case_name[1]
    y_values_dict = defaultdict(list)

    x_label = "NO X LABEL"
    x_values = list()

    yvalues = list()
    runtime = list()
    for item in case_results:

      found_cycle = False
      found_flops = False

      if "radius_cm" in item:
        # Radii were multiplied by 10 to avoid casting the value to int
        # while storing as JSON. Check and fix.
        x_label = 'Radius [m]'
        x_values.append(item["radius_cm"] / 100.0)
      elif "num_points" in item:
        x_label = 'Number of points'
        x_values.append(item["num_points"])
      else:
        print("No problem size values found, using bar plot!")

      if "preallocated_memory_B" in item:
        if not "preallocated_memory_B" in y_values_dict:
          y_values_dict["preallocated_memory_B"] = list()
        y_values_dict["preallocated_memory_B"].append(item["preallocated_memory_B"] / 1e6)

      if "preallocated_blocks" in item:
        if not "preallocated_blocks" in y_values_dict:
          y_values_dict["preallocated_blocks"] = list()
        y_values_dict["preallocated_blocks"].append(item["preallocated_blocks"])

      if "flops" in item:
        if not "flops" in y_values_dict:
          y_values_dict["flops"] = list()
        y_values_dict["flops"].append(item["flops"])
        found_flop = True

      if "cpu_time" in item:
        if not "cpu_time" in y_values_dict:
          y_values_dict["cpu_time"] = list()
          y_values_dict["cycles"] = list()
        runtime_seconds = item["cpu_time"] * \
            helpers.UnitToScaler(item["time_unit"])
        y_values_dict["cpu_time"].append(runtime_seconds)
        y_values_dict["cycles"].append(runtime_seconds * cpu_freq_to_use_mhz * 1e6)
        found_cycle = True

      if found_flop and found_cycle:
        if not "flops_p_cycle" in y_values_dict:
          y_values_dict["flops_p_cycle"] = list()
        assert "flops" in y_values_dict
        assert "cycles" in y_values_dict
        assert len(y_values_dict["flops"]) == len(y_values_dict["cycles"])
        y_values_dict["flops_p_cycle"].append(
            float(y_values_dict["flops"][-1]) / y_values_dict["cycles"][-1])

    print(x_label)
    print(x_values)

    print(y_label_dict)
    print(y_values_dict)

    # If we have no problem size parameter, i.e. no range for x, use bar plot.
    if not x_values:

      for y_values_key, y_values in y_values_dict.iteritems():
        assert len(y_values) == 1
        if y_values_key is "flops_p_cycle":
          ax.bar(bar_indent[split_case_name[1]], y_values[0], bar_width,
                 color=colors[split_case_name[1]], label=split_case_name[1])
        elif y_values_key is "cpu_time":
          axr.bar(bar_indent[split_case_name[1]], y_values[0],
                  bar_width, color=colors[split_case_name[1]], label=split_case_name[1])
        elif y_values_key is "flops":
          axf.bar(bar_indent[split_case_name[1]], y_values[0],
                  bar_width, color=colors[split_case_name[1]], label=split_case_name[1])
        elif y_values_key is "preallocated_memory_B":
          axm.bar(bar_indent[split_case_name[1]], y_values[0],
                  bar_width, color=colors[split_case_name[1]], label=split_case_name[1])
        elif y_values_key is "cycles":
          axc.bar(bar_indent[split_case_name[1]], y_values[0],
                  bar_width, color=colors[split_case_name[1]], label=split_case_name[1])
        elif y_values_key is "preallocated_blocks":
          # Don't plot
          print("skip preallocated_blocks")
        else:
          sys.exit("Unknown y values for bar plot")

      benchmark_name = str.split(str(benchmark_data[0][0]["name"]), '/')[0]

      # ax.yaxis.grid(True, linestyle='-', color='white')
      title = 'Performance for ' + benchmark_name
      ax.patch.set_facecolor('#E2E2E2')
      ax.set_title(title)
      ax.set_ylabel(y_label_dict["flops_p_cycle"])
      ax.legend(loc=0)
      ax.set_xlim([0, 1.0])

      # axr.yaxis.grid(True, linestyle='-', color='white')
      axr.patch.set_facecolor('#E2E2E2')
      title = 'Runtime for ' + benchmark_name
      axr.set_title(title)
      axr.set_ylabel(y_label_dict["cpu_time"])
      axr.legend(loc=0)
      axr.set_xlim([0, 1.0])

      # axm.yaxis.grid(True, linestyle='-', color='white')
      axm.patch.set_facecolor('#E2E2E2')
      title = 'Memory for ' + benchmark_name
      axm.set_title(title)
      axm.set_ylabel(y_label_dict["preallocated_memory_B"])
      axm.legend(loc=0)
      axm.set_xlim([0, 1.0])

      # axf.yaxis.grid(True, linestyle='-', color='white')
      axf.patch.set_facecolor('#E2E2E2')
      title = 'Flops for ' + benchmark_name
      axf.set_title(title)
      axf.set_ylabel(y_label_dict["flops"])
      axf.legend(loc=0)
      axf.set_xlim([0, 1.0])

      # axc.yaxis.grid(True, linestyle='-', color='white')
      axc.patch.set_facecolor('#E2E2E2')
      title = 'Cycles for ' + benchmark_name
      axc.set_title(title)
      axc.set_ylabel(y_label_dict["cycles"])
      axc.legend(loc=0)
      axc.set_xlim([0, 1.0])

    # If we have a range for the x axis, use line plot.
    else:
      for y_values_key, y_values in y_values_dict.iteritems():
        if y_values_key is "flops_p_cycle":
          ax.plot(x_values, y_values, marker='o', markeredgecolor='none',
                  color=colors[split_case_name[1]], linewidth=2, markersize=6, label=split_case_name[1])
        elif y_values_key is "cpu_time":
          axr.plot(x_values, y_values, marker='o', markeredgecolor='none',
                   color=colors[split_case_name[1]], linewidth=2, markersize=6, label=split_case_name[1])
        elif y_values_key is "flops":
          axf.plot(x_values, y_values, marker='o', markeredgecolor='none',
                   color=colors[split_case_name[1]], linewidth=2, markersize=6, label=split_case_name[1])
        elif y_values_key is "preallocated_memory_B":
          axm.plot(x_values, y_values, marker='o', markeredgecolor='none',
                   color=colors[split_case_name[1]], linewidth=2, markersize=6, label=split_case_name[1])
        elif y_values_key is "cycles":
          axc.plot(x_values, y_values, marker='o', markeredgecolor='none',
                   color=colors[split_case_name[1]], linewidth=2, markersize=6, label=split_case_name[1])
        elif y_values_key is "preallocated_blocks":
          # Don't plot
          print("skip preallocated_blocks")
        else:
          sys.exit("Unknown y values for line plot")

        benchmark_name = str.split(str(benchmark_data[0][0]["name"]), '/')[0]

        ax.patch.set_facecolor('#E2E2E2')
        ax.yaxis.grid(True, linestyle='-', color='white')
        title = 'Performance for ' + benchmark_name
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label_dict["flops_p_cycle"])
        ax.legend(loc=0)

        axr.patch.set_facecolor('#E2E2E2')
        axr.yaxis.grid(True, linestyle='-', color='white')
        title = 'Runtime for ' + benchmark_name
        axr.set_title(title)
        axr.set_xlabel(x_label)
        axr.set_ylabel(y_label_dict["cpu_time"])
        axr.legend(loc=0)

        axm.patch.set_facecolor('#E2E2E2')
        axm.yaxis.grid(True, linestyle='-', color='white')
        title = 'Memory for ' + benchmark_name
        axm.set_title(title)
        axm.set_xlabel(x_label)
        axm.set_ylabel(y_label_dict["preallocated_memory_B"])
        axm.legend(loc=0)

        axf.patch.set_facecolor('#E2E2E2')
        axf.yaxis.grid(True, linestyle='-', color='white')
        title = 'Flops for ' + benchmark_name
        axf.set_title(title)
        axf.set_xlabel(x_label)
        axf.set_ylabel(y_label_dict["flops"])
        axf.legend(loc=0)

        axc.patch.set_facecolor('#E2E2E2')
        axc.yaxis.grid(True, linestyle='-', color='white')
        title = 'Cycles for ' + benchmark_name
        axc.set_title(title)
        axc.set_xlabel(x_label)
        axc.set_ylabel(y_label_dict["cycles"])
        axc.legend(loc=0)

  for key, figures_to_save in figures_dict.items():
    figures_to_save.savefig(key + ".performance.pdf")
  for key, figures_to_save in figures_flops_dict.items():
    figures_to_save.savefig(key + ".flops.pdf")
  for key, figures_to_save in figures_cycles_dict.items():
    figures_to_save.savefig(key + ".cycles.pdf")
  for key, figures_to_save in figures_runtime_dict.items():
    figures_to_save.savefig(key + ".runtime.pdf")
  for key, figures_to_save in figures_memory_dict.items():
    figures_to_save.savefig(key + ".memory.pdf")

  return figures_dict.items()

# Generate a plot for each benchmark task within this report.


def GeneratePlotsForBenchmarkFile(filename, real_cpu_freq_mhz):
  print 'Generating plots for benchmark file: ' + filename
  json_data = LoadGoogleBenchmarkJsonReportResults(filename)
  benchmark_context = json_data["context"]

  figures = list()
  results = defaultdict(list)
  for benchmark_data in json_data["benchmarks"]:
    name_string = str(benchmark_data['name'])
    split_name = str.split(name_string, '/')
    if len(split_name) > 2:
      split_name = split_name[0:2]

    benchmark, case = split_name  # str.split(name_string, '/')
    results[case].append(benchmark_data)

  results_by_case = list()
  for key, value in results.items():
    results_by_case.append(value)

  fig = GeneratePerformancePlotOverParameters(benchmark_context, results_by_case, real_cpu_freq_mhz)
  figures.append(fig)

  return figures


# Parse input arguments.
parser = argparse.ArgumentParser(description='Lifelong calibration evaluation.')
parser.add_argument('--voxblox-workspace', dest='voxblox_workspace', nargs='?',
                    default="/home/user/code/htwfnc_ws", help='Voxblox workspace.')
parser.add_argument('--show-on-screen', dest='show_on_screen', type=bool,
                    default=True, help='Show plots on screen?')
parser.add_argument('--real-cpu-frequency_mhz', dest='real_cpu_freq_mhz', type=int,
                    default=-1, help='Actual CPU frequency, needed if turbo boost is turned off, bceause Google benchmark does not write the correct frequency into the benchmark results.')
parsed = parser.parse_args()

# Build, run the benchmarks and collect the results.
assert(os.path.isdir(parsed.voxblox_workspace))
# helpers.RunAllBenchmarksOfPackage(parsed.voxblox_workspace, "htwfsc_benchmarks")
benchmark_files = helpers.GetAllBenchmarkingResultsOfPackage(
    parsed.voxblox_workspace, "htwfsc_benchmarks")

# Generate a plot for each benchmark result file.
figures = list()
for benchmark_file in benchmark_files:
  figures.append(GeneratePlotsForBenchmarkFile(benchmark_file, parsed.real_cpu_freq_mhz))

if parsed.show_on_screen:
  plt.show()
