#!/bin/bash
cd /home/jbarnard/deepjets
source /data/edawe/public/setup.sh

path=/coepp/cephfs/mel/edawe/deepjets/events/pythia/raw

python maxout_file_args.py $path/w_events_j1p0_sj0p50_jets_images.h5 \
                           $path/qcd_events_j1p0_sj0p50_jets_images.h5 \
                           datasets/w_mwin_1-0p5 models/w_mwin_1-0p5 &
python maxout_file_args.py $path/w_events_j1p0_sj0p50_jets_zoomed_images.h5 \
                           $path/qcd_events_j1p0_sj0p50_jets_zoomed_images.h5 \
                           datasets/w_mwin_1-0p5_z models/w_mwin_1-0p5_z &
python maxout_file_args.py $path/w_events_j1p0_sj0p50_jets_shrink_images.h5 \
                           $path/qcd_events_j1p0_sj0p50_jets_shrink_images.h5 \
                           datasets/w_mwin_1-0p5_sz models/w_mwin_1-0p5_sz &

python maxout_file_args.py $path/w_events_j1p0_sj0p30_jets_images.h5 \
                           $path/qcd_events_j1p0_sj0p30_jets_images.h5 \
                           datasets/w_mwin_1-0p3 models/w_mwin_1-0p3 &
python maxout_file_args.py $path/w_events_j1p0_sj0p30_jets_zoomed_images.h5 \
                           $path/qcd_events_j1p0_sj0p30_jets_zoomed_images.h5 \
                           datasets/w_mwin_1-0p3_z models/w_mwin_1-0p3_z &
python maxout_file_args.py $path/w_events_j1p0_sj0p30_jets_shrink_images.h5 \
                           $path/qcd_events_j1p0_sj0p30_jets_shrink_images.h5 \
                           datasets/w_mwin_1-0p3_sz models/w_mwin_1-0p3_sz &

python maxout_file_args.py $path/w_events_j1p2_sj0p20_jets_images.h5 \
                           $path/qcd_events_j1p2_sj0p20_jets_images.h5 \
                           datasets/w_mwin_1p2-0p2 models/w_mwin_1p2-0p2 &
python maxout_file_args.py $path/w_events_j1p2_sj0p20_jets_zoomed_images.h5 \
                           $path/qcd_events_j1p2_sj0p20_jets_zoomed_images.h5 \
                           datasets/w_mwin_1p2-0p2_z models/w_mwin_1p2-0p2_z &
python maxout_file_args.py $path/w_events_j1p2_sj0p20_jets_shrink_images.h5 \
                           $path/qcd_events_j1p2_sj0p20_jets_shrink_images.h5 \
                           datasets/w_mwin_1p2-0p2_sz models/w_mwin_1p2-0p2_sz &