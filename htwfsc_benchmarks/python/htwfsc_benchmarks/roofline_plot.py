#!/usr/bin/env pyhton
import argparse
from label_lines import *
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Gill Sans MT']
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def PlotRoofline(title, arch_flops_per_cycle, arch_memory_bandwidth, 
                 baseline_pts, optimized_pts):
    # Plot the roof.
    opint_start = 1e-4
    opint_intersection = arch_flops_per_cycle / arch_memory_bandwidth
    opint_max = 1e3

    colors = {'Baseline': '#284910', 'Fast': '#C51929', 'Other': '#ECC351'}

    x = np.linspace(opint_intersection, opint_max, 1000)
    y_perf = arch_flops_per_cycle * np.ones(x.size)
    fig, ax = plt.subplots()
    label='performance ' + str(arch_flops_per_cycle) + ' flops/cycle'
    plt.loglog(x, y_perf, label=label, lw=2, color='k')
    ax.annotate(label, (opint_intersection + 5, arch_flops_per_cycle + 0.5))
    
    x2 = np.linspace(opint_start, opint_intersection)
    y_oi = arch_memory_bandwidth * x2
    label='memory bandwidth ' + str(arch_memory_bandwidth) + ' bytes/cycle'
    plt.loglog(x2, y_oi, label=label, lw=2, color='k')
    ax.annotate(label, (2.0 * opint_start, arch_flops_per_cycle / 22.0))
    
    # Plot the benchmarking points.
    all_opint=list()
    all_perf=list()
    for param_value, opint, perf in baseline_pts:
        all_opint.append(opint)
        all_perf.append(perf)
    plt.loglog(all_opint, all_perf, 'o-', color=colors['Baseline'], label='baseline', 
               linewidth=2, markersize=6)
    ax.annotate(baseline_pts[0][0], (all_opint[0], all_perf[0]))
    ax.annotate(baseline_pts[-1][0], (all_opint[-1], all_perf[-1]))
        
    all_opint=list()
    all_perf=list()
    for param_value, opint, perf in optimized_pts:
        all_opint.append(opint)
        all_perf.append(perf)
    plt.loglog(all_opint, all_perf, 'o-', color=colors['Fast'], label='Fast', 
               linewidth=2, markersize=6)
    ax.annotate(optimized_pts[0][0], (all_opint[0], all_perf[0]))
    ax.annotate(optimized_pts[-1][0], (all_opint[-1], all_perf[-1]))
    
    ax.set_xlabel('operational intensity [flops/byte]')
    ax.set_ylabel('performance [flops/cycle]')
    ax.grid(True)
    
    ax.patch.set_facecolor('#E2E2E2')
    baseline_patch = mpatches.Patch(color='r', label='Baseline')
    optimized_patch = mpatches.Patch(color='g', label='Fast')
    plt.legend([mpatches.Patch(color=colors['Baseline']),
                mpatches.Patch(color=colors['Fast'])], 
                ['Baseline','Fast'], loc=4)

    ax.set_ylim([0, 2 * arch_flops_per_cycle])
            
    plt.tight_layout()
    plt.title(title)
    return fig

#############################################3
# CPU data Skylake; SSE only not other stuff.
arch_flops_per_cycle = 8 # 2 mul_ps / cycle * 4 float / mul_ps
arch_memory_bandwidth = 3.71 
arch_cpu_freq_hz = 2.8e9

# Roofline E2E with varying radius
# (param_value, opint, performance)
baseline_pts = [
#     Elapsed Time:    1.635s
#     CPU Time:    1.617s
#     Memory Bound:    14.9%
#     Loads:    495014850
#     Stores:    620409306
#     LLC Miss Count:    1,800,108
#     Average Latency (cycles):    10
#     Total Thread Count:    4
#     Paused Time:    0s
#     E2EBenchmark/Radius_Baseline/300  161395733 ns  160893220 ns         12 flops=9.25365M preallocated_blocks=7.8125k preallocated_memory_B=375.32M radius_cm=150
    ( 300,  
     (9.25365 * 1e6) / ((495014850 + 620409306) / 12), 
     (9.25365 * 1e6) / ((161395733 / 1e9) * arch_cpu_freq_hz)),

#     Elapsed Time:    3.301s
#     CPU Time:    3.271s
#     Memory Bound:    22.9%
#     Loads:    1461043830
#     Stores:    2008830132
#     LLC Miss Count:    2,400,144
#     Average Latency (cycles):    11
#     Total Thread Count:    4
#     Paused Time:    0s
#     E2EBenchmark/Radius_Baseline/500  253233569 ns  253212600 ns         12 flops=14.9964M preallocated_blocks=32k preallocated_memory_B=1.50128G radius_cm=250
    ( 500,  
     (14.9964 * 1e6) / ((1461043830 + 2008830132) / 12), 
     (14.9964 * 1e6) / ((253233569/ 1e9) * arch_cpu_freq_hz)),
               
#     Elapsed Time:    6.213s
#     CPU Time:    6.134s
#     Memory Bound:    27.5%
#     Loads:    2990489712
#     Stores:    6093691404
#     LLC Miss Count:    6,300,378
#     Average Latency (cycles):    10
#     Total Thread Count:    3
#     Paused Time:    0s
#     E2EBenchmark/Radius_Baseline/700  348537543 ns  347659215 ns          8 flops=20.7429M preallocated_blocks=83.1875k preallocated_memory_B=3.90275G radius_cm=350
    ( 700,  
     (20.7429 * 1e6) / ((2990489712 + 6093691404) / 8), 
     (20.7429 * 1e6) / ((348537543 / 1e9) * arch_cpu_freq_hz)),

#     Elapsed Time:    12.325s
#     CPU Time:    12.195s
#     Memory Bound:    27.2%
#     Loads:    6481394436
#     Stores:    13694605416
#     LLC Miss Count:    11,700,702
#     Average Latency (cycles):    12
#     Total Thread Count:    4
#     Paused Time:    0s     
#     E2EBenchmark/Radius_Baseline/900  454538800 ns  454510315 ns          8 flops=26.4791M preallocated_blocks=190.539k preallocated_memory_B=8.93915G radius_cm=450
    ( 900,  
     (26.4791 * 1e6) / ((6481394436 + 13694605416) / 8), 
     (26.4791 * 1e6) / ((454538800 / 1e9) * arch_cpu_freq_hz)),
    
#     Elapsed Time:    11.924s
#     CPU Time:    11.802s
#     Memory Bound:    26.8%
#     Loads:    7036411086
#     Stores:    12801792024
#     LLC Miss Count:    12,600,756
#     Average Latency (cycles):    10
#     Total Thread Count:    4
#     Paused Time:    0s           
#     E2EBenchmark/Radius_Baseline/1100  538687557 ns  538567361 ns          4 flops=32.2382M preallocated_blocks=334.961k preallocated_memory_B=15.7147G radius_cm=550
    (1100,  
     (32.2382 * 1e6) / ((7036411086 + 12801792024) / 4), 
     (32.2382 * 1e6) / ((538687557 / 1e9) * arch_cpu_freq_hz)),
]
print baseline_pts

optimized_pts = [
#     Elapsed Time:    1.240s
#     CPU Time:    1.224s
#     Memory Bound:    21.0%
#     Loads:    1165834974
#     Stores:    1011615174
#     LLC Miss Count:    2,100,126
#     Average Latency (cycles):    10
#     Total Thread Count:    4
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Fast/300    9547280 ns    9544420 ns         62 flops=1.92017M num_points=299 preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    ( 300,  
     (1.92017 * 1e6) / ((1165834974 + 1011615174) / 62),
     (1.92017 * 1e6) / ((9547280 / 1e9) * arch_cpu_freq_hz)),
    
#     Elapsed Time:    1.433s
#     CPU Time:    1.417s
#     Memory Bound:    17.3%
#     Loads:    1343440302
#     Stores:    1062015930
#     LLC Miss Count:    4,500,270
#     Average Latency (cycles):    14
#     Total Thread Count:    3
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Fast/500   15400707 ns   15399492 ns         45 flops=3.18794M num_points=502 preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    ( 500,  
     (3.18794 * 1e6) / ((1343440302 + 1062015930) / 45),
     (3.18794 * 1e6) / ((15400707 / 1e9) * arch_cpu_freq_hz)),
    
#     Elapsed Time:    1.601s
#     CPU Time:    1.582s
#     Memory Bound:    19.3%
#     Loads:    1779053370
#     Stores:    1198817982
#     LLC Miss Count:    3,600,216
#     Average Latency (cycles):    14
#     Total Thread Count:    4
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Fast/700   22245160 ns   22230833 ns         30 flops=4.4859M num_points=702 preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    ( 700,  
     (4.4859 * 1e6) / ((1779053370 + 1198817982) / 30),
     (4.4859 * 1e6) / ((22245160 / 1e9) * arch_cpu_freq_hz)),
   
#     Elapsed Time:    1.783s
#     CPU Time:    1.759s
#     Memory Bound:    17.9%
#     Loads:    1563046890
#     Stores:    1176017640
#     LLC Miss Count:    4,200,252
#     Average Latency (cycles):    13
#     Total Thread Count:    4
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Fast/900   27693864 ns   27692123 ns         26 flops=5.7316M num_points=900 preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    ( 900,  
     (5.7316 * 1e6) / ((1563046890 + 1176017640) / 26),
     (5.7316 * 1e6) / ((27693864/ 1e9) * arch_cpu_freq_hz)),
   
#     Elapsed Time:    1.874s
#     CPU Time:    1.843s
#     Memory Bound:    11.1%
#         L1 Bound:    5.4%
#         DRAM Bound:    
#             DRAM Bandwidth Bound:    27.7%
#             LLC Miss:    11.5%
#     Loads:    1674650238
#     Stores:    1212018180
#     LLC Miss Count:    3,300,198
#     Average Latency (cycles):    9
#     Total Thread Count:    4
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Fast/1100   34268889 ns   34266724 ns         17 flops=7.01916M num_points=1099 preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    (1100,  
     (7.01916 * 1e6) / ((1674650238 + 1212018180) / 17),
     (7.01916 * 1e6) / ((34268889/ 1e9) * arch_cpu_freq_hz))
]
print optimized_pts

title = 'Roofline for varying radius of the ball [m]'
fig = PlotRoofline(title, arch_flops_per_cycle, arch_memory_bandwidth, 
                   baseline_pts, optimized_pts)
fig.savefig("roofline-over-num-pts.pdf", bbox_inches='tight')


# Roofline E2E with varying num_pts
# (param_value, opint, performance)
baseline_pts = [
#     Elapsed Time:    1.125s
#     CPU Time:    1.116s
#     Memory Bound:    15.8%
#     Loads:    576617298
#     Stores:    177602664
#     LLC Miss Count:    600,036
#     Average Latency (cycles):    12
#     Total Thread Count:    4
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Baseline/100   11291838 ns   11290853 ns         60 flops=649.234k num_points=100 preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    (100,  
     (649.234 * 1e3) / ((576617298 + 177602664) / 60),
     (649.234 * 1e3) / ((11291838/ 1e9) * arch_cpu_freq_hz)),
                
#     Elapsed Time:    1.358s
#     CPU Time:    1.345s
#     Memory Bound:    13.5%
#     Loads:    686420592
#     Stores:    597608964
#     LLC Miss Count:    900,054
#     Average Latency (cycles):    11
#     Total Thread Count:    4
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Baseline/1000  111416452 ns  111408442 ns          7 flops=6.36184M num_points=1000 preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    (1000,  
     (6.36184 * 1e6) / ((686420592 + 597608964) / 7),
     (6.36184 * 1e6) / ((111416452/ 1e9) * arch_cpu_freq_hz)),
    
#     Elapsed Time:    2.368s
#     CPU Time:    2.350s
#     Memory Bound:    21.0%
#     Loads:    891626748
#     Stores:    693610404
#     LLC Miss Count:    900,054
#     Average Latency (cycles):    10
#     Total Thread Count:    3
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Baseline/10000 1082054141 ns 1081976441 ns          1 flops=63.7697M num_points=9.76367k preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    (10000,  
     (63.7697 * 1e6) / ((891626748 + 693610404) / 1),
     (63.7697 * 1e6) / ((1082054141 / 1e9) * arch_cpu_freq_hz)),
    
#     Elapsed Time:    22.136s
#     CPU Time:    21.910s
#     Memory Bound:    12.2%
#     Loads:    8646259380
#     Stores:    4774871622
#     LLC Miss Count:    10,500,630
#     Average Latency (cycles):    8
#     Total Thread Count:    3
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Baseline/100000 10939030867 ns 10919051961 ns          1 flops=638.115M num_points=97.6602k preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    (100000,  
     (638.115 * 1e6) / ((8646259380 + 4774871622) / 1),
     (638.115 * 1e6) / ((10939030867 / 1e9) * arch_cpu_freq_hz)),
    
#     Elapsed Time:    215.559s
#     CPU Time:    202.621s
#     Memory Bound:    10.5%
#     Loads:    83681510370
#     Stores:    42529837938
#     LLC Miss Count:    72,904,374
#     Average Latency (cycles):    9
#     Total Thread Count:    4
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Baseline/1000000 107362518926 ns 107169169568 ns          1 flops=6.23069G num_points=976.561k preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    (1000000,  
     (6.23069 * 1e9) / ((83681510370 + 42529837938) / 1),
     (6.23069 * 1e9) / ((107362518926 / 1e9) * arch_cpu_freq_hz)),
]
print baseline_pts

optimized_pts = [
#     Elapsed Time:    1.427s
#     CPU Time:    1.408s
#     Memory Bound:    14.4%
#     Loads:    1650049500
#     Stores:    1177217658
#     LLC Miss Count:    600,036
#     Average Latency (cycles):    13
#     Total Thread Count:    3
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Fast/100    2673009 ns    2672842 ns        254 flops=647.359k num_points=100 preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    (100,  
     (647.359 * 1e3) / ((1650049500 + 1177217658) / 254),
     (647.359 * 1e3) / ((2673009 / 1e9) * arch_cpu_freq_hz)),
           
#     Elapsed Time:    1.800s
#     CPU Time:    1.779s
#     Memory Bound:    24.3%
#         L1 Bound:    8.9%
#         DRAM Bound:    
#             DRAM Bandwidth Bound:    0.6%
#             LLC Miss:    9.8%
#     Loads:    1617048510
#     Stores:    1100416506
#     LLC Miss Count:    2,700,162
#     Average Latency (cycles):    12
#     Total Thread Count:    3
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Fast/1000   30014550 ns   30009494 ns         23 flops=6.40606M num_points=1000 preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    (1000,  
     (6.40606 * 1e6) / ((1617048510 + 1100416506) / 23),
     (6.40606 * 1e6) / ((30014550 / 1e9) * arch_cpu_freq_hz)),
           
#     Elapsed Time:    3.425s
#     CPU Time:    3.392s
#     Memory Bound:    14.5%
#     Loads:    2527875834
#     Stores:    1411221168
#     LLC Miss Count:    3,000,180
#     Average Latency (cycles):    10
#     Total Thread Count:    3
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Fast/10000  281637523 ns  281613526 ns          2 flops=63.7323M num_points=9.76367k preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    (10000,  
     (63.7323 * 1e6) / ((2527875834 + 1411221168) / 2),
     (63.7323 * 1e6) / ((281637523 / 1e9) * arch_cpu_freq_hz)),
    
#     Elapsed Time:    13.976s
#     CPU Time:    13.860s
#     Memory Bound:    15.2%
#     Loads:    8671460136
#     Stores:    4376465646
#     LLC Miss Count:    9,600,576
#     Average Latency (cycles):    11
#     Total Thread Count:    4
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Fast/100000 2788449554 ns 2787536182 ns          1 flops=637.997M num_points=97.6602k preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    (100000,  
     (637.997 * 1e6) / ((8671460136 + 4376465646) / 1),
     (637.997 * 1e6) / ((2788449554 / 1e9) * arch_cpu_freq_hz)),
     
#     Elapsed Time:    135.601s
#     CPU Time:    134.371s
#     Memory Bound:    10.0%
#     Loads:    83889116598
#     Stores:    42011430162
#     LLC Miss Count:    82,204,932
#     Average Latency (cycles):    10
#     Total Thread Count:    3
#     Paused Time:    0s
#     E2EBenchmark/NumPoints_Fast/1000000 27586165106 ns 27584027812 ns          1 flops=6.23034G num_points=976.561k preallocated_blocks=2.67969k preallocated_memory_B=128.735M
    (1000000,  
     (6.23034 * 1e9) / ((83889116598 + 42011430162) / 1),
     (6.23034 * 1e9) / ((27586165106 / 1e9) * arch_cpu_freq_hz)),                 
]
print optimized_pts

title = 'Roofline for varying number of points on the ball [-]'
fig = PlotRoofline(title, arch_flops_per_cycle, arch_memory_bandwidth, 
                   baseline_pts, optimized_pts)
fig.savefig("roofline-over-radius.pdf", bbox_inches='tight')

plt.show()
