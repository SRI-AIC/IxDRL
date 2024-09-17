import argparse
import json
import logging
import os

from ixdrl import Rollouts
from ixdrl.analysis import get_interestingness_dataframe, print_stats
from ixdrl.analysis.full import FullAnalysis
from ixdrl.util.cmd_line import str2bool, save_args, str2log_level
from ixdrl.util.io import create_clear_dir, load_object, save_object
from ixdrl.util.logging import change_log_handler

__author__ = 'Pedro Sequeira'
__email__ = 'pedro.sequeira@sri.com'
__description__ = 'Performs several analyses of interestingness over previously-collected interaction data.'

INTERESTINGNESS_DATA_FILE = 'interestingness.pkl.gz'
INTERESTINGNESS_CSV_FILE = 'interestingness.csv.gz'
INTERESTINGNESS_PANDAS_FILE = 'interestingness_pd.pkl.gz'
INTERESTINGNESS_PLOTS_FILE = 'interestingness_plots.pkl.gz'


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Pickle file containing the interaction data collected using `bin.collect.*` scripts.')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Directory in which to save results')
    parser.add_argument('--img-format', '-if', type=str, default='pdf', help='The format of image files.')
    parser.add_argument('--derivative_accuracy', '-da', type=int, default=4,
                        help='The accuracy used by finite difference methods. Needs to be a positive, even number.')
    parser.add_argument('--processes', '-p', type=int, default=-1,
                        help='The number of parallel processes to use for this analysis. A value of `-1` or `None` '
                             'will use all available CPUs.')
    parser.add_argument('--clear', '-c', type=str2bool, help='Clear output directories before generating results.')
    parser.add_argument('--verbosity', '-v', type=str2log_level, default=0, help='Verbosity level.')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        raise ValueError('Could not find interaction data file in {}'.format(args.input))

    # check output dir and log file
    out_dir = args.output
    create_clear_dir(out_dir, args.clear)
    save_args(args, os.path.join(out_dir, 'args.json'))
    change_log_handler(os.path.join(out_dir, 'interestingness.log'), args.verbosity)

    # load interaction data
    logging.info(f'Loading interaction data from: {args.input}...')
    interaction_data: Rollouts = load_object(args.input)
    logging.info(f'\tLoaded data for {len(interaction_data)} rollouts')

    # create and run full analysis and saves results
    analysis = FullAnalysis(interaction_data, args.derivative_accuracy, args.processes)
    results = analysis.analyze()
    dimensions = list(results.keys())
    logging.info(f'Finished analyzing interestingness from {len(interaction_data)} rollouts.')
    logging.info(f'{len(dimensions)} dimensions extracted:\n{dimensions}')
    with open(os.path.join(out_dir, 'dimensions.json'), 'w') as fp:
        json.dump(dimensions, fp, indent=4)

    # save consolidated results
    data_file = os.path.join(out_dir, INTERESTINGNESS_DATA_FILE)
    save_object(results, data_file, compress_gzip=True)
    logging.info(f'Saved results to "{data_file}"')

    # get dataframe for all dimensions and episodes, save to CSV and pandas file
    df = get_interestingness_dataframe(results, interaction_data)
    data_file = os.path.join(out_dir, INTERESTINGNESS_CSV_FILE)
    df.to_csv(data_file, index=False)
    logging.info(f'Saved results to CSV file: "{data_file}"')

    data_file = os.path.join(out_dir, INTERESTINGNESS_PANDAS_FILE)
    df.to_pickle(data_file)
    logging.info(f'Saved results to Pandas pickle file: "{data_file}"')

    # generate plots of the different analyses
    stats_dir = os.path.join(out_dir, 'stats')
    create_clear_dir(stats_dir, args.clear)
    logging.info(f'Saving interestingness statistics to {stats_dir}...')
    figures = print_stats(df, stats_dir, [ia.name for ia in analysis], args.img_format)
    figs_file = os.path.join(out_dir, INTERESTINGNESS_PLOTS_FILE)
    save_object(figures, figs_file, compress_gzip=True)
    logging.info(f'Saved all plotly figures to "{figs_file}"')

    logging.info('Done!')


if __name__ == '__main__':
    main()
