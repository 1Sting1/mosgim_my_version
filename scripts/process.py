"""
Process TEC data from various sources and create ionospheric maps.

This script handles the processing pipeline for TEC data:
1. Loading data from different sources (HDF5, TXT, RINEX)
2. Processing and filtering the data
3. Converting to magnetic coordinates
4. Solving for spherical harmonic coefficients
5. Creating and saving ionospheric maps
"""

import os
import argparse
import time
import numpy as np

from pathlib import Path
from datetime import datetime, timedelta

from mosgim.data.tec_prepare import (DataSourceType,
                                       MagneticCoordType,
                                       ProcessingType,
                                       process_data,
                                       combine_data,
                                       get_data,
                                       save_data,
                                       sites,
                                       calculate_seed_mag_coordinates_parallel)
from mosgim.data.loader import (LoaderHDF, 
                                LoaderTxt)
from mosgim.mosg.map_creator import (solve_weights,
                                calculate_maps)
from mosgim.mosg.lcp_solver import create_lcp
from mosgim.plotter.animation import plot_and_save


def populate_output_paths(args: argparse.Namespace) -> None:
    """
    Populate output file paths if not specified in arguments.
    
    Args:
        args: Command line arguments
    """
    date = args.date
    mag_type = args.mag_type
    output_path = args.out_path
    
    if output_path:
        if not args.modip_file:
            args.modip_file = output_path / f'prepared_mdip_{date}.npz'
        if not args.mag_file:
            args.mag_file = output_path / f'prepared_mag_{date}.npz'
        if not args.weight_file:
            args.weight_file = output_path / f'weights_{mag_type}_{date}.npz'
        if not args.lcp_file:
            args.lcp_file = output_path / f'lcp_{mag_type}_{date}.npz'
        if not args.maps_file:
            args.maps_file = output_path / f'maps_{mag_type}_{date}.npz'
        if not args.animation_file:
            args.animation_file = output_path / f'animation_{mag_type}_{date}.mp4'


def parse_arguments(command: str = '') -> Iterator[argparse.Namespace]:
    """
    Parse command line arguments.
    
    Args:
        command: Optional command string for testing
        
    Returns:
        Iterator of parsed arguments for each day to process
    """
    parser = argparse.ArgumentParser(
        description='Process TEC data from various sources (HDF5, TXT, RINEX)'
    )
    
    # Required arguments
    parser.add_argument(
        '--data_path',
        type=Path,
        required=True,
        help='Path to input data directory'
    )
    parser.add_argument(
        '--process_type',
        type=ProcessingType,
        required=True,
        choices=list(ProcessingType),
        help='Type of processing: single day or date range'
    )
    parser.add_argument(
        '--data_source',
        type=DataSourceType,
        required=True,
        choices=list(DataSourceType),
        help='Source data format'
    )
    parser.add_argument(
        '--date',
        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--mag_type',
        type=MagneticCoordType,
        required=True,
        choices=list(MagneticCoordType),
        help='Type of magnetic coordinates'
    )
    
    # Optional arguments
    parser.add_argument(
        '--out_path',
        type=Path,
        default=Path('/tmp/'),
        help='Output directory path'
    )
    parser.add_argument(
        '--ndays',
        type=int,
        help='Number of days to process for ranged processing'
    )
    parser.add_argument(
        '--modip_file',
        type=Path,
        help='Path for modified dip angle results'
    )
    parser.add_argument(
        '--mag_file',
        type=Path,
        help='Path for magnetic latitude results'
    )
    parser.add_argument(
        '--weight_file',
        type=Path,
        help='Path for solved weights'
    )
    parser.add_argument(
        '--lcp_file',
        type=Path,
        help='Path for LCP results'
    )
    parser.add_argument(
        '--nsite',
        type=int,
        help='Number of sites to process'
    )
    parser.add_argument(
        '--nworkers',
        type=int,
        default=1,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--memory_per_worker',
        type=int,
        default=2,
        help='Memory limit per worker in GB'
    )
    parser.add_argument(
        '--skip_prepare',
        action='store_true',
        help='Skip data preparation, use existing files'
    )
    parser.add_argument(
        '--animation_file',
        type=Path,
        help='Path for animation output'
    )
    parser.add_argument(
        '--maps_file',
        type=Path,
        help='Path for map data output'
    )
    parser.add_argument(
        '--const',
        action='store_true',
        help='Use constant instead of linear interpolation'
    )

    args = parser.parse_args(command.split()) if command else parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out_path, exist_ok=True)
    if args.process_type == ProcessingType.ranged and (args.ndays is None):
        parser.error("Ranged processing requires --ndays")
        
    # Generate arguments for each day
    if args.process_type == ProcessingType.ranged:
        base_time = args.date
        for day_offset in range(args.ndays):
            current_args = argparse.Namespace(**vars(args))
            current_date = base_time + timedelta(day_offset)
            day_of_year = str(current_date.timetuple().tm_yday).zfill(3)
            current_args.date = current_date
            current_args.data_path = f"{args.data_path}/{current_date.year}/{day_of_year}"
            populate_output_paths(current_args)
            yield current_args
    else:
        pass


def process_data_for_date(args: argparse.Namespace) -> None:
    """
    Process TEC data for a single date.
    
    Args:
        args: Command line arguments for the date
    """
    print(f"Processing data for {args.date}")
    start_time = time.time()
    
    if not args.skip_prepare:
        if args.nsite:
            _sites = sites[:args.nsite]
        else:
            _sites = sites[:]
        if args.data_source == DataSourceType.hdf:
            loader = LoaderHDF(args.data_path)
            data_generator = loader.generate_data(sites=_sites)

        elif args.data_source == DataSourceType.txt:
            loader = LoaderTxt(args.data_path)
            data_generator = loader.generate_data_pool(sites=_sites, 
                                                       nworkers=args.nworkers)
        else:
            raise ValueError('Define data source')
        data = process_data(data_generator)
        print(f"Sites not found: {loader.not_found_sites}")
        print(f"Data reading completed in {time.time() - start_time:.2f}s")
        
        # Process data in parallel
        print("Starting magnetic coordinate calculations...")
        start_time = time.time()
        data_chunks = combine_data(data, nchunks=args.nworkers)
        print('Start magnetic calculations...')
        st = time.time()
        result = calculate_seed_mag_coordinates_parallel(data_chunks, 
                                                        nworkers=args.nworkers)
        print(f'Done, took {time.time() - st}')
        
        # Save processed data
        if args.mag_file and args.modip_file:
            save_data(result, args.modip_file, args.mag_file, args.date)
            
        data = get_data(result, args.mag_type, args.date)
    else:
        if args.mag_type == MagneticCoordType.mag:
            data = np.load(args.mag_file, allow_pickle=True)
        elif args.mag_type == MagneticCoordType.mdip:
            data = np.load(args.modip_file, allow_pickle=True)
    weights, N = solve_weights(data, 
                               nworkers=args.nworkers, 
                               gigs=args.memory_per_worker,
                               linear= not args.const)
    if args.weight_file:
        np.savez(args.weight_file, res=weights, N=N)
        
    # Create LCP solution
    try:
        lcp_result = create_lcp({'res': weights, 'N': N})
    except Exception as e:
        print(f'LCP calculation failed: {e}')
        return
        
    if args.lcp_file:
        np.savez(args.lcp_file, res=lcp_result, N=N)
        
    # Calculate and save maps
    maps = calculate_maps(lcp_result, args.mag_type, args.date)
    plot_and_save(maps, args.animation_file, args.maps_file)


if __name__ == '__main__':
    for args in parse_arguments():
        process_data_for_date(args)
    
