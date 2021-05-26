#!/usr/bin/env python2

import os
import argparse
from ruamel.yaml import YAML
import shutil
import json
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from colorama import init, Fore

import add_path
from trajectory import Trajectory
import plot_utils as pu
import results_writer as res_writer
from analyze_trajectory_single import analyze_multiple_trials
from fn_constants import kNsToEstFnMapping, kNsToMatchFnMapping, kFnExt

init(autoreset=True)

rc('font', **{'family': 'serif', 'serif': ['Cardo']})
rc('text', usetex=True)

FORMAT = '.pdf'

PALLETE = ['b', 'g', 'r', 'c', 'k', 'y', 'm']

def plot_trajectories(dataset_trajectories_list, dataset_names, algorithm_names,
                      datasets_out_dir, plot_settings, plot_idx=0, plot_side=True,
                      plot_aligned=True, plot_traj_per_alg=True):
    for dataset_idx, dataset_nm in enumerate(dataset_names):
        output_dir = datasets_out_dir[dataset_nm]
        dataset_trajs = dataset_trajectories_list[dataset_idx]
        p_es_0 = {}
        p_gt_raw = (dataset_trajs[0])[plot_idx].p_gt_raw
        p_gt_0 = {}
        for traj_list in dataset_trajs:
            p_es_0[traj_list[plot_idx].alg] = traj_list[plot_idx].p_es_aligned
            p_gt_0[traj_list[plot_idx].alg] = traj_list[plot_idx].p_gt
        print("Collected trajectories to plot: {0}".format(algorithm_names))
        assert sorted(algorithm_names) == sorted(list(p_es_0.keys()))

        print("Plotting {0}...".format(dataset_nm))

        # plot trajectory
        fig = plt.figure(figsize=(6, 5.5))
        ax = fig.add_subplot(111, aspect='equal',
                             xlabel='x [m]', ylabel='y [m]')
        if dataset_nm in plot_settings['datasets_titles']:
            ax.set_title(plot_settings['datasets_titles'][dataset_nm])
        for alg in algorithm_names:
            pu.plot_trajectory_top(ax, p_es_0[alg],
                                   plot_settings['algo_colors'][alg],
                                   plot_settings['algo_labels'][alg])
        plt.sca(ax)
        # pu.plot_trajectory_top(ax, p_gt_raw, 'm', 'Groundtruth')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.tight_layout()
        fig.savefig(output_dir+'/' + dataset_nm +
                    '_trajectory_top'+FORMAT, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)

        # plot trajectory side
        if not plot_side:
            continue
        fig = plt.figure(figsize=(6, 2.2))
        ax = fig.add_subplot(111, aspect='equal',
                             xlabel='x [m]', ylabel='z [m]')
        if dataset_nm in plot_settings['datasets_titles']:
            ax.set_title(plot_settings['datasets_titles'][dataset_nm])
        for alg in algorithm_names:
            pu.plot_trajectory_side(ax, p_es_0[alg],
                                    plot_settings['algo_colors'][alg],
                                    plot_settings['algo_labels'][alg])
        plt.sca(ax)
        # pu.plot_trajectory_side(ax, p_gt_raw, 'm', 'Groundtruth')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig.tight_layout()
        fig.savefig(output_dir+'/'+dataset_nm +
                    '_trajectory_side'+FORMAT, bbox_inches="tight", dpi=args.dpi)
        plt.close(fig)

def parse_config_file(config_fn, sort_names):
    yaml = YAML()
    with open(config_fn) as f:
        d = yaml.load(f)
    datasets = d['Datasets'].keys()
    if sort_names:
        datasets.sort()
    datasets_labels = {}
    datasets_titles = {}
    for v in datasets:
        datasets_labels[v] = d['Datasets'][v]['label']
        if 'title' in d['Datasets'][v]:
            datasets_titles[v] = d['Datasets'][v]['title']

    algorithms = d['Algorithms'].keys()
    if sort_names:
        algorithms.sort()
    alg_labels = {}
    alg_fn = {}
    for v in algorithms:
        alg_labels[v] = d['Algorithms'][v]['label']
        alg_fn[v] = d['Algorithms'][v]['fn']

    boxplot_distances = []
    if 'RelDistances' in d:
        boxplot_distances = d['RelDistances']
    boxplot_percentages = []
    if 'RelDistancePercentages' in d:
        boxplot_percentages = d['RelDistancePercentages']

    if boxplot_distances and boxplot_percentages:
        print(Fore.RED + "Found both both distances and percentages for boxplot distances")
        print(Fore.RED + "Will use the distances instead of percentages.")
        boxplot_percentages = []

    return datasets, datasets_labels, datasets_titles, algorithms, alg_labels, alg_fn,\
        boxplot_distances, boxplot_percentages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Analyze trajectories''')

    default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '../results')
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '../analyze_trajectories_config')

    parser.add_argument('config', type=str,
                        help='yaml file specifying algorithms and datasets')
    parser.add_argument(
        '--output_dir',
        help="Folder to output plots and data",
        default=default_path)
    parser.add_argument(
        '--results_dir', help='base folder with the results to analyze',
        default=default_path)

    parser.add_argument(
        '--platform', help='HW platform: [laptop, nuc, odroid, up]',
        default='laptop')
    parser.add_argument(
        '--mul_trials', type=int,
        help='number of trials, None for single run', default=None)
    parser.add_argument('--no_sort_names', action='store_false', dest='sort_names',
                        help='whether to sort dataset and algorithm names')

    # plot trajectories
    parser.add_argument('--plot_trajectories',
                        help='Plot the trajectories', action='store_true')
    parser.add_argument('--no_plot_side', action='store_false', dest='plot_side')
    parser.add_argument('--no_plot_aligned', action='store_false', dest='plot_aligned')
    parser.add_argument('--no_plot_traj_per_alg', action='store_false',
                        dest='plot_traj_per_alg')

    parser.add_argument('--recalculate_errors',
                        help='Deletes cached errors', action='store_true')
    parser.add_argument('--png',
                        help='Save plots as png instead of pdf',
                        action='store_true')
    parser.add_argument('--dpi', type=int, default=300)
    parser.set_defaults(odometry_error_per_dataset=False, overall_odometry_error=False,
                        rmse_table=False,
                        plot_trajectories=False, rmse_boxplot=False,
                        recalculate_errors=False, png=False, time_statistics=False,
                        sort_names=True, plot_side=True, plot_aligned=True,
                        plot_traj_per_alg=True, rmse_median_only=False)
    args = parser.parse_args()
    print("Arguments:\n{}".format(
        '\n'.join(['- {}: {}'.format(k, v)
                   for k, v in args.__dict__.items()])))

    print("Will analyze results from {0} and output will be "
          "in {1}".format(args.results_dir, args.output_dir))
    output_dir = args.output_dir

    config_fn = os.path.join(config_path, args.config)

    print("Parsing evaluation configuration {0}...".format(config_fn))

    datasets, datasets_labels, datasets_titles,\
        algorithms, algo_labels, algo_fn, rel_e_distances, rel_e_perc = \
        parse_config_file(config_fn, args.sort_names)
    shutil.copy2(config_fn, output_dir)
    datasets_res_dir = {}
    for d in datasets:
        cur_res_dir = os.path.join(output_dir, '{}_{}_results'.format(args.platform, d))
        if os.path.exists(cur_res_dir):
            shutil.rmtree(cur_res_dir)
        os.makedirs(cur_res_dir)
        datasets_res_dir[d] = cur_res_dir
    same_subtraj = True if rel_e_distances else False
    assert len(PALLETE) > len(algorithms),\
        "Not enough colors for all configurations"
    algo_colors = {}
    for i in range(len(algorithms)):
        algo_colors[algorithms[i]] = PALLETE[i]

    print(Fore.YELLOW+"=== Evaluation Configuration Summary ===")
    print(Fore.YELLOW+"Datasests to evaluate: ")
    for d in datasets:
        print(Fore.YELLOW+'- {0}: {1}'.format(d, datasets_labels[d]))
    print(Fore.YELLOW+"Algorithms to evaluate: ")
    for a in algorithms:
        print(Fore.YELLOW+'- {0}: {1}, {2}, {3}'.format(a, algo_labels[a],
                                                        algo_fn[a],
                                                        algo_colors[a]))
    plot_settings = {'datasets_labels': datasets_labels,
                     'datasets_titles': datasets_titles,
                     'algo_labels': algo_labels,
                     'algo_colors': algo_colors}

    if args.png:
        FORMAT = '.png'

    eval_uid = '_'.join(list(plot_settings['algo_labels'].values())) +\
        datetime.now().strftime("%Y%m%d%H%M")

    n_trials = 1
    if args.mul_trials:
        print(Fore.YELLOW +
              "We will ananlyze multiple trials #{0}".format(args.mul_trials))
        n_trials = args.mul_trials

    need_odometry_error = args.odometry_error_per_dataset or args.overall_odometry_error
    if need_odometry_error:
        print(Fore.YELLOW+"Will calculate odometry errors")

    print("#####################################")
    print(">>> Loading and calculating errors....")
    print("#####################################")
    # organize by configuration
    config_trajectories_list = []
    config_multierror_list = []
    dataset_boxdist_map = {}
    for d in datasets:
        dataset_boxdist_map[d] = rel_e_distances
    for config_i in algorithms:
        cur_trajectories_i = []
        cur_mulierror_i = []
        for d in datasets:
            print(Fore.RED + "--- Processing {0}-{1}... ---".format(
                config_i, d))
            exp_name = args.platform + '_' + config_i + '_' + d
            trace_dir = os.path.join(args.results_dir,
                                     args.platform, config_i, exp_name)
            assert os.path.exists(trace_dir),\
                "{0} not found.".format(trace_dir)
            traj_list, mt_error = analyze_multiple_trials(
                trace_dir, algo_fn[config_i], n_trials,
                recalculate_errors=args.recalculate_errors,
                preset_boxplot_distances=dataset_boxdist_map[d])
            if not dataset_boxdist_map[d] and traj_list:
                print("Assign the boxplot distances for {0}...".format(d))
                dataset_boxdist_map[d] = traj_list[0].preset_boxplot_distances
            for traj in traj_list:
                traj.alg = config_i
                traj.dataset_short_name = d
            mt_error.saveErrors()
            mt_error.uid = '_'.join([args.platform, config_i,
                                     d, str(n_trials)])
            mt_error.alg = config_i
            mt_error.dataset = d
            cur_trajectories_i.append(traj_list)
            cur_mulierror_i.append(mt_error)
        config_trajectories_list.append(cur_trajectories_i)
        config_multierror_list.append(cur_mulierror_i)

    # organize by dataset name
    dataset_trajectories_list = []
    dataset_multierror_list = []
    for ds_idx, dataset_nm in enumerate(datasets):
        dataset_trajs = [v[ds_idx] for v in config_trajectories_list]
        dataset_trajectories_list.append(dataset_trajs)
        dataset_multierrors = [v[ds_idx] for v in config_multierror_list]
        dataset_multierror_list.append(dataset_multierrors)

    if args.plot_trajectories:
        print(Fore.MAGENTA+'--- Plotting trajectory top and side view ... ---')
        plot_trajectories(dataset_trajectories_list, datasets, algorithms,
                          datasets_res_dir, plot_settings, plot_side=args.plot_side,
                          plot_aligned=args.plot_aligned,
                          plot_traj_per_alg=args.plot_traj_per_alg)

    import subprocess as s
    s.call(['notify-send', 'rpg_trajectory_evaluation finished',
            'results in: {0}'.format(os.path.abspath(output_dir))])
    print("#####################################")
    print("<<< Finished.")
    print("#####################################")
