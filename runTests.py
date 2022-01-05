#!/usr/bin/env python

import argparse
import csv
import datetime
import logging
import os
import re
import subprocess
import sys

## Functions to parse output of benchmarks, specifically, to get the
## running time.

# Default parser
def DefaultParseOutput(output):
    for line in output.split('\n'):
        # print("LINE:" + line.lower())
        m = re.match(r"\D*time\D+(\d+\.\d+)", line.lower())
        if m:
            return m.group(1)
    logger.warning("No time output found")
    return str(0.0)

# Output parser for lavaMD, sradv1
def TimeNewLineParseOutput(output):
    foundTimeMarker = False
    for line in output.split('\n'):
        # print(line)
        if foundTimeMarker:
            m = re.match(r"(\d+\.\d+)", line)
            return m.group(1)
        m = re.match(r"Total time:", line)
        if m:
            foundTimeMarker = True
    logger.warning("No time output found")
    return str(0.0)

# lud benchmark outputs running time in milliseconds, rather than
# seconds.
def ludParseOutput(output):
    timeMS = DefaultParseOutput(output)
    return str(float(timeMS) / 1000.0)

# particlefilter benchmark outputs running time in an unusual way
def particlefilterParseOutput(output):
    for line in output.split('\n'):
        # print("LINE:" + line.lower())
        m = re.match(r"ENTIRE PROGRAM TOOK\D+(\d+\.\d+)", line)
        if m:
            return m.group(1)
    logger.warning("No time output found")
    return str(0.0)

benchmarks = {
    # "b+tree": ["", "./b+tree.out core 2 file ../../data/b+tree/mil.txt command ../../data/b+tree/command.txt"],
    # "backprop": ["", "./backprop 1048576"],
    "bfs": ("bfs", "./bfs 4 ../../data/bfs/graph1MW_6.txt", DefaultParseOutput),
    "cfd": ("cfd", "./euler3d_cpu ../../data/cfd/fvcorr.domn.193K", DefaultParseOutput),
    # "heartwall": ["heartwall", "./heartwall ../../data/heartwall/test.avi 20 4"],
    "hotspot": ("hotspot", "./hotspot 8192 8192 2 4 ../../data/hotspot/temp_8192 ../../data/hotspot/power_8192 output.out", DefaultParseOutput),
    "hotspot3D": ("hotspot3D", "./3D 512 8 100 ../../data/hotspot3D/power_512x8 ../../data/hotspot3D/temp_512x8 output.out", DefaultParseOutput),
    "kmeans": ("kmeans/kmeans_cilk", "./kmeans -n 1024 -i ../../../data/kmeans/kdd_cup", DefaultParseOutput),
    "lavaMD": ("lavaMD", "./lavaMD -cores 4 -boxes1d 10", TimeNewLineParseOutput),
    "leukocyte": ("leukocyte/Cilk", "./leukocyte 5 4 ../../../data/leukocyte/testfile.avi", DefaultParseOutput),
    "lud": ("lud/cilk", "./lud_cilk -s 8000", ludParseOutput),
    "nn": ("nn", "./nn filelist_4 5 30 90", DefaultParseOutput),
    "nw": ("nw", "./needle 2048 10 2", DefaultParseOutput),
    "particlefilter": ("particlefilter", "./particle_filter -x 128 -y 128 -z 10 -np 10000", particlefilterParseOutput),
    # "pathfinder": ["pathfinder", "./pathfinder 100000 100 > out"],
    "sradv1": ("srad/srad_v1", "./srad 100 0.5 502 458 4", TimeNewLineParseOutput),
    "sradv2": ("srad/srad_v2", "./srad 2048 2048 0 127 0 127 2 0.5 2", DefaultParseOutput),
    "streamcluster": ("streamcluster", "./sc_cilk 10 20 256 65536 65536 1000 none output.txt 4", DefaultParseOutput)
}

parallelSystems = {
    "opencilk" : { "build": ["cilk", "CILKFLAG=\"-fopencilk\"", "", "", "CILK_NWORKERS"],
                   "benchparams" : {
                       "kmeans": { "dir": "kmeans/kmeans_cilk" },
                       "leukocyte": { "dir": "leukocyte/Cilk" },
                       "lud": { "dir": "lud/cilk", "runcmd" : "./lud_cilk -s 8000" }
                       }
                   },
    "cilkplus" : { "build": ["cilk", "CILKFLAG=\"-fcilkplus\"", "",
                             "EXTRA_LDFLAGS=\"-L/data/compilers/cilkrts/build -Wl,-rpath,/data/compilers/cilkrts/build\"",
                             "CILK_NWORKERS"],
                   "benchparams" : {
                       "kmeans": { "dir": "kmeans/kmeans_cilk" },
                       "leukocyte": { "dir": "leukocyte/Cilk" },
                       "lud": { "dir": "lud/cilk", "runcmd" : "./lud_cilk -s 8000" }
                       }
                   },
    "openmp" : { "build" : ["openmp", "", "", "", "OMP_NUM_THREADS"],
                 "benchparams" : {
                       "kmeans": { "dir": "kmeans/kmeans_openmp" },
                       "leukocyte": { "dir": "leukocyte/OpenMP" },
                       "lud": { "dir": "lud/omp", "runcmd" : "./lud_omp -s 8000" },
                       "streamcluster": { "runcmd" : "./sc_omp 10 20 256 65536 65536 1000 none output.txt 4"}
                       }
                   }
}

defaultWorkerCounts = [1,2,4,8,12,16,24,32,40,48]
defaultNumTrials = 5

compilerBinDir = "/data/animals/opencilk/install/bin/"

logger = logging.getLogger(__name__)

def compileTest(systemFlag, extraCFlags, extraLDFlags):
    subProcCommand = "make clean; make CC=" + compilerBinDir + "clang CXX=" + compilerBinDir + "clang++" + " " + systemFlag + " " + extraCFlags + " " + extraLDFlags
    logger.info(subProcCommand)
    proc = subprocess.Popen([subProcCommand], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,err = proc.communicate()
    # Log the output to stdout and stderr from running the command.
    logger.info("STDOUT:\n" + str(out, "utf-8"))
    logger.info("STDERR:\n" + str(err, "utf-8"))

def runTest(workerCountEnvVar, command, workerCounts, numTrials, parseOutputFn):
    results = []
    for P in workerCounts:
        subProcCommand = [workerCountEnvVar + '=' + str(P) + ' taskset -c 0-' + str(P-1) + ' ' + command]
        logger.info(subProcCommand)
        trials = [str(P)]
        for t in range(0,numTrials):
            proc = subprocess.Popen(subProcCommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out,err = proc.communicate()
            # Log the output to stdout and stderr from running the command.
            logger.info("STDOUT:\n" + str(out, "utf-8"))
            logger.info("STDERR:\n" + str(err, "utf-8"))
            trials.append(parseOutputFn(str(out, "utf-8")))
        results.append(trials)
    return results

def buildAndRun(csvWriter, benchmarkList, systemList, workerCounts, numTrials):
    basedir = os.path.dirname(__file__)

    # Iterate over listed benchmarks
    for bench in benchmarkList:
        if bench not in benchmarks:
            logger.warning("Unknown benchmark: " + bench)
            continue
        for system in systemList:
            if system not in parallelSystems.keys():
                logger.warning("Unknown system: " + system)
                continue
            sysbuild = parallelSystems[system]["build"]
            sysbenchparams = parallelSystems[system]["benchparams"]
            sysdir = sysbuild[0]
            if bench in sysbenchparams:
                if "dir" in sysbenchparams[bench]:
                    subdir = sysbenchparams[bench]["dir"]
                else:
                    subdir = benchmarks[bench][0]
                if "runcmd" in sysbenchparams[bench]:
                    runcmd = sysbenchparams[bench]["runcmd"]
                else:
                    runcmd = benchmarks[bench][1]
            else:
                subdir = benchmarks[bench][0]
                runcmd = benchmarks[bench][1]
            os.chdir(os.path.join(os.path.join(basedir, sysdir), subdir))
            logger.info("TEST: " + bench + ", " + system)
            # Compile the test
            compileTest(sysbuild[1], sysbuild[2], sysbuild[3])
            # Run the test and parse the output
            results = runTest(sysbuild[4], runcmd, workerCounts, numTrials, benchmarks[bench][2])
            # Log the result in CSV form
            for trials in results:
                csvWriter.writerow([bench, system] + trials)

# Log the status of turboboost
def logTurboboostStatus():
    subProcCommand = "cat /sys/devices/system/cpu/intel_pstate/no_turbo"
    proc = subprocess.Popen([subProcCommand], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out,err = proc.communicate()
    logger.info(subProcCommand + ":" + str(out, "utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", "-b", help="comma-separated list of benchmarks to run")
    ap.add_argument("--systems", "-s", help="comma-separated list of parallel-computing systems to run")
    ap.add_argument("--output", "-o", help="tag for output filename, or '-' to dump to stdout")
    ap.add_argument("--trials", "-x", help="number of trials to run for each test")
    ap.add_argument("--workers", "-w", help="comma-separated list of worker counts to test")

    args = ap.parse_args()
    print(args)

    # Set output
    if args.output == '-':
        logging.basicConfig(level=logging.INFO)
        csvOutput = None
    else:
        logname = "results_" + datetime.datetime.now().strftime("%Y%m%d-%H%M")
        if args.output is not None:
            logname = logname + "_" + args.output
        csvOutput = logname + ".csv"
        logname = logname + ".out"
        logging.basicConfig(filename=logname, level=logging.INFO)

    # Set list of benchmarks to run
    if args.benchmarks is None:
        benchmarkList = list(benchmarks.keys())
    else:
        benchmarkList = list(args.benchmarks.split(","))

    print(benchmarkList)

    # Set list of systems to test
    if args.systems is None:
        systemList = list(parallelSystems.keys())
    else:
        systemList = []
        for system in args.systems.split(","):
            if system not in parallelSystems:
                logger.warning("Unknown system: " + system)
                continue
            systemList.append(system)

    # Set number of trials to run
    if args.trials is None:
        numTrials = defaultNumTrials
    else:
        numTrials = int(args.trials)

    # Set number of workers to test
    if args.workers is None:
        workerCounts = defaultWorkerCounts
    else:
        workerCounts = list(map(int, args.workers.split(",")))

    logTurboboostStatus()
    if csvOutput is None:
        buildAndRun(csv.writer(sys.stdout), benchmarkList, systemList, workerCounts, numTrials)
    else:
        with open(csvOutput, 'w', newline='') as csvFile:
            buildAndRun(csv.writer(csvFile), benchmarkList, systemList, workerCounts, numTrials)

    logger.info("All tests complete.")

if __name__ == '__main__':
    main()
