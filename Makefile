# simple makefile to simplify repetitive build env management tasks under posix

PYTHON := $(shell which python)
NOSETESTS := $(shell which nosetests)

output := /coepp/cephfs/mel/edawe/deepjets/events
setup := cd /data/edawe/private/deepjets_dev; source setup.sh;
WMASS := 80.385

.PHONY: events

all: clean inplace

clean-pyc:
	@find . -name "*.pyc" -exec rm {} \;

clean-so:
	@find deepjets -name "*.so" -exec rm {} \;

clean-build:
	@rm -rf build

clean: clean-build clean-pyc clean-so

in: inplace # just a shortcut
inplace:
	@$(PYTHON) setup.py build_ext -i

test: inplace
	@$(NOSETESTS) -s -v deepjets

test-events:
	mkdir -p $(output)/log
	# default samples used in http://arxiv.org/abs/1511.05190
	#echo "$(setup) ./generate wprime.config --events 500000 --output $(output)/wprime_default_0p6.h5 --cut-on-pdgid 24 --pt-min 250 --pt-max 300 --jet-size 0.6" | qsub -e $(output)/log -o $(output)/log -N wprime_default_0p6
	echo "$(setup) ./generate w.config --events 500000 --output $(output)/w_default_0p6.h5 --params \"PhaseSpace:pTHatMin = 230;PhaseSpace:pTHatMax = 320\" --trimmed-pt-min 250 --trimmed-pt-max 300 --jet-size 0.6" | qsub -e $(output)/log -o $(output)/log -N w_default_0p6
	echo "$(setup) ./generate qcd.config --events 500000 --output $(output)/qcd_default_0p6.h5 --params \"PhaseSpace:pTHatMin = 230;PhaseSpace:pTHatMax = 320\" --trimmed-pt-min 250 --trimmed-pt-max 300 --jet-size 0.6" | qsub -e $(output)/log -o $(output)/log -N qcd_default_0p6
	# no shrink
	for pthatmin in $$(seq 150 50 500); do \
		echo "$(setup) ./generate qcd.config --jet-size 1.2 --events 100000 --output $(output)/qcd_1p2_$${pthatmin}.h5 --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N qcd_1p2_$${pthatmin}; \
		echo "$(setup) ./generate w.config --jet-size 1.2 --events 100000 --output $(output)/w_1p2_$${pthatmin}.h5 --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N w_1p2_$${pthatmin}; \
		echo "$(setup) ./generate qcd.config --jet-size 1 --events 100000 --output $(output)/qcd_1p0_$${pthatmin}.h5 --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N qcd_1p0_$${pthatmin}; \
		echo "$(setup) ./generate w.config --jet-size 1 --events 100000 --output $(output)/w_1p0_$${pthatmin}.h5 --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N w_1p0_$${pthatmin}; \
	done
	# shrink with jet mass
	for pthatmin in $$(seq 150 50 500); do \
		echo "$(setup) ./generate qcd.config --jet-size 1.2 --events 100000 --output $(output)/qcd_shrink_1p2_$${pthatmin}.h5 --shrink --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N qcd_shrink_1p2_$${pthatmin}; \
		echo "$(setup) ./generate w.config --jet-size 1.2 --events 100000 --output $(output)/w_shrink_1p2_$${pthatmin}.h5 --shrink --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N w_shrink_1p2_$${pthatmin}; \
		echo "$(setup) ./generate qcd.config --jet-size 1 --events 100000 --output $(output)/qcd_shrink_1p0_$${pthatmin}.h5 --shrink --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N qcd_shrink_1p0_$${pthatmin}; \
		echo "$(setup) ./generate w.config --jet-size 1 --events 100000 --output $(output)/w_shrink_1p0_$${pthatmin}.h5 --shrink --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N w_shrink_1p0_$${pthatmin}; \
	done
	# shrink with W mass
	for pthatmin in $$(seq 150 50 500); do \
		echo "$(setup) ./generate qcd.config --jet-size 1.2 --events 100000 --output $(output)/qcd_shrink_wmass_1p2_$${pthatmin}.h5 --shrink --shrink-mass $(WMASS) --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N qcd_shrink_wmass_1p2_$${pthatmin}; \
		echo "$(setup) ./generate w.config --jet-size 1.2 --events 100000 --output $(output)/w_shrink_wmass_1p2_$${pthatmin}.h5 --shrink --shrink-mass $(WMASS) --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N w_shrink_wmass_1p2_$${pthatmin}; \
		echo "$(setup) ./generate qcd.config --jet-size 1 --events 100000 --output $(output)/qcd_shrink_wmass_1p0_$${pthatmin}.h5 --shrink --shrink-mass $(WMASS) --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N qcd_shrink_wmass_1p0_$${pthatmin}; \
		echo "$(setup) ./generate w.config --jet-size 1 --events 100000 --output $(output)/w_shrink_wmass_1p0_$${pthatmin}.h5 --shrink --shrink-mass $(WMASS) --seed $${pthatmin} --params \"PhaseSpace:pTHatMin = $${pthatmin}\"" | qsub -e $(output)/log -o $(output)/log -N w_shrink_wmass_1p0_$${pthatmin}; \
	done
	echo "$(setup) ./generate w.config --events 100000 --output $(output)/w_test.h5 --params \"PhaseSpace:pTHatMin = 250;PhaseSpace:pTHatMax = 300\" --shrink " | qsub -e $(output)/log -o $(output)/log -N w_test
	echo "$(setup) ./generate qcd.config --events 100000 --output $(output)/qcd_test.h5 --params \"PhaseSpace:pTHatMin = 250;PhaseSpace:pTHatMax = 300\" --shrink " | qsub -e $(output)/log -o $(output)/log -N qcd_test

events:
	mkdir -p $(output)/log
	for chunk in $$(seq 1 1 10); do \
		echo "$(setup) ./generate w.config --output $(output)/w_events_shrink_$${chunk}.h5 --shrink --events 10000 --random-state $${chunk} --trimmed-mass-min 65 --trimmed-mass-max 95 weighted" | qsub -e $(output)/log -o $(output)/log -N w_events_shrink_$${chunk} -l nodes=1:ppn=1 -q long; \
	done
	for chunk in $$(seq 1 1 10); do \
		echo "$(setup) ./generate qcd.config --output $(output)/qcd_events_shrink_$${chunk}.h5 --shrink --events 10000 --random-state $${chunk} --trimmed-mass-min 65 --trimmed-mass-max 95 weighted" | qsub -e $(output)/log -o $(output)/log -N qcd_events_shrink_$${chunk} -l nodes=1:ppn=1 -q long; \
	done

events-no-shrink:
	mkdir -p $(output)/log
	for chunk in $$(seq 1 1 10); do \
		echo "$(setup) ./generate w.config --output $(output)/w_events_noshrink_$${chunk}.h5 --events 10000 --random-state $${chunk} --trimmed-mass-min 65 --trimmed-mass-max 95 weighted" | qsub -e $(output)/log -o $(output)/log -N w_events_noshrink_$${chunk} -l nodes=1:ppn=1 -q long; \
	done
	for chunk in $$(seq 1 1 10); do \
		echo "$(setup) ./generate qcd.config --output $(output)/qcd_events_noshrink_$${chunk}.h5 --events 10000 --random-state $${chunk} --trimmed-mass-min 65 --trimmed-mass-max 95 weighted" | qsub -e $(output)/log -o $(output)/log -N qcd_events_noshrink_$${chunk} -l nodes=1:ppn=1 -q long; \
	done

events-stanford:
	mkdir -p $(output)/log
	for chunk in $$(seq 1 1 10); do \
		echo "$(setup) ./generate w.config --output $(output)/w_events_stanford_$${chunk}.h5 --events 100000 --random-state $${chunk} --trimmed-mass-min 65 --trimmed-mass-max 95 --params \"PhaseSpace:pTHatMin = 230;PhaseSpace:pTHatMax = 320\" --jet-size 0.6 unweighted --trimmed-pt-min 250 --trimmed-pt-max 300" | qsub -e $(output)/log -o $(output)/log -N w_events_stanford_$${chunk} -l nodes=1:ppn=1 -q long; \
	done
	for chunk in $$(seq 1 1 10); do \
		echo "$(setup) ./generate qcd.config --output $(output)/qcd_events_stanford_$${chunk}.h5 --events 100000 --random-state $${chunk} --trimmed-mass-min 65 --trimmed-mass-max 95 --params \"PhaseSpace:pTHatMin = 230;PhaseSpace:pTHatMax = 320\" --jet-size 0.6 unweighted --trimmed-pt-min 250 --trimmed-pt-max 300" | qsub -e $(output)/log -o $(output)/log -N qcd_events_stanford_$${chunk} -l nodes=1:ppn=1 -q long; \
	done

all-events: events events-no-shrink events-stanford

raw-events:
	mkdir -p $(output)/log
	for chunk in $$(seq 1 1 14); do \
		echo "$(setup) ./generate w.config --output $(output)/w_events_$${chunk}.h5 --events 1000000 --random-state $${chunk} --params \"PhaseSpace:pTHatMin = 150;PhaseSpace:pTHatMax = 550\"" | qsub -e $(output)/log -o $(output)/log -N w_events_$${chunk} -l nodes=1:ppn=1; \
	done
	for chunk in $$(seq 1 1 14); do \
		echo "$(setup) ./generate qcd.config --output $(output)/qcd_events_$${chunk}.h5 --events 1000000 --random-state $${chunk} --params \"PhaseSpace:pTHatMin = 150;PhaseSpace:pTHatMax = 550\"" | qsub -e $(output)/log -o $(output)/log -N qcd_events_$${chunk} -l nodes=1:ppn=1; \
	done

w-images-no-batch:
	mkdir -p $(output)/log
	for chunk in $$(seq 1 1 10); do \
		./generate-images w.config $(output)/w_images_noshrink2.h5 --events 10000 --jobs 10 --random-state $${chunk}; \
	done

qcd-images-no-batch:
	mkdir -p $(output)/log
	for chunk in $$(seq 1 1 10); do \
		./generate-images qcd.config $(output)/qcd_images_noshrink2.h5 --events 10000 --jobs 10 --random-state $${chunk}; \
	done

sherwig:
	#./generate --events 100000 --shrink /data/edawe/public/deepjets/events/herwig/hepmc/QCD_Angular.hepmc --output /data/edawe/public/deepjets/events/herwig/QCD_Angular.h5 &
	#./generate --events 100000 --shrink /data/edawe/public/deepjets/events/herwig/hepmc/QCD_Dipole.hepmc --output /data/edawe/public/deepjets/events/herwig/QCD_Dipole.h5 &
	#./generate --events 100000 --shrink /data/edawe/public/deepjets/events/herwig/hepmc/WDecay_Angular.hepmc --output /data/edawe/public/deepjets/events/herwig/WDecay_Angular.h5 &
	#./generate --events 100000 --shrink /data/edawe/public/deepjets/events/herwig/hepmc/WDecay_Dipole2.hepmc --output /data/edawe/public/deepjets/events/herwig/WDecay_Dipole2.h5 &
	./generate --events 100000 --shrink /data/edawe/public/deepjets/events/sherpa/JZ/jZ_Events.hepmc2g --output /data/edawe/public/deepjets/events/sherpa/jZ_Events.h5 &
	./generate --events 100000 --shrink /data/edawe/public/deepjets/events/sherpa/WZ/WZ_Events.hepmc2g --output /data/edawe/public/deepjets/events/sherpa/WZ_Events.h5 &


pythia:
	$(eval output_path := $(output)/pythia/images)
	$(eval queue := long)
	$(eval events := 100000)
	$(eval params := PhaseSpace:pTHatMin = 230;PhaseSpace:pTHatMax = 320)
	$(eval seed_offset := 100000)
	mkdir -p $(output_path)/log
	# default
	for seed in $$(seq $(seed_offset) 1 $$(($(seed_offset) + 10))); do \
		chunk=`printf "%04d" $${seed}`; \
		./generate --batch $(queue) --random-state $${seed} w.config --events $(events) --output $(output_path)/w_$${chunk}.h5 --params "$(params)"; \
	done
	for seed in $$(seq $(seed_offset) 1 $$(($(seed_offset) + 10))); do \
		chunk=`printf "%04d" $${seed}`; \
		./generate --batch $(queue) --random-state $${seed} qcd.config --events $(events) --output $(output_path)/qcd_$${chunk}.h5 --params "$(params)"; \
	done

pythia-vincia:
	$(eval output_path := $(output)/pythia/images)
	$(eval queue := long)
	$(eval events := 100000)
	$(eval params := PhaseSpace:pTHatMin = 230;PhaseSpace:pTHatMax = 320)
	$(eval seed_offset := 100000)
	mkdir -p $(output_path)/log
	# VINCIA
	for seed in $$(seq $(seed_offset) 1 $$(($(seed_offset) + 10))); do \
		chunk=`printf "%04d" $${seed}`; \
		./generate --batch $(queue) --random-state $${seed} w_vincia.config --shower vincia --events $(events) --output $(output_path)/w_vincia_$${chunk}.h5 --params "$(params)"; \
	done
	for seed in $$(seq $(seed_offset) 1 $$(($(seed_offset) + 10))); do \
		chunk=`printf "%04d" $${seed}`; \
		./generate --batch $(queue) --random-state $${seed} qcd_vincia.config --shower vincia --events $(events) --output $(output_path)/qcd_vincia_$${chunk}.h5 --params "$(params)"; \
	done

pythia-dire:
	$(eval output_path := $(output)/pythia/images)
	$(eval queue := long)
	$(eval events := 100000)
	$(eval params := PhaseSpace:pTHatMin = 230;PhaseSpace:pTHatMax = 320)
	$(eval seed_offset := 100000)
	mkdir -p $(output_path)/log
	# DIRE
	for seed in $$(seq $(seed_offset) 1 $$(($(seed_offset) + 10))); do \
		chunk=`printf "%04d" $${seed}`; \
		./generate --batch $(queue) --random-state $${seed} w_dire.config --shower dire --events $(events) --output $(output_path)/w_dire_$${chunk}.h5 --params "$(params)"; \
	done
	for seed in $$(seq $(seed_offset) 1 $$(($(seed_offset) + 10))); do \
		chunk=`printf "%04d" $${seed}`; \
		./generate --batch $(queue) --random-state $${seed} qcd_dire.config --shower dire --events $(events) --output $(output_path)/qcd_dire_$${chunk}.h5 --params "$(params)"; \
	done

pythia-uncert:
	$(eval output_path := $(output)/pythia/images/uncert)
	$(eval queue := long)
	$(eval events := 100000)
	$(eval params := PhaseSpace:pTHatMin = 230;PhaseSpace:pTHatMax = 320;UncertaintyBands:doVariations = off)
	$(eval seed_offset := 100000)
	mkdir -p $(output_path)/log
	# mu=0.5
	for seed in $$(seq $(seed_offset) 1 $$(($(seed_offset) + 20))); do \
		chunk=`printf "%04d" $${seed}`; \
		./generate --batch $(queue) --random-state $${seed} w.config --events $(events) --output $(output_path)/w_murfac05_$${chunk}.h5 --params "$(params);SigmaProcess:renormMultFac = 0.5"; \
	done
	for seed in $$(seq $(seed_offset) 1 $$(($(seed_offset) + 50))); do \
		chunk=`printf "%04d" $${seed}`; \
		./generate --batch $(queue) --random-state $${seed} qcd.config --events $(events) --output $(output_path)/qcd_murfac05_$${chunk}.h5 --params "$(params);SigmaProcess:renormMultFac = 0.5"; \
	done
	# mu=2.0
	for seed in $$(seq $(seed_offset) 1 $$(($(seed_offset) + 20))); do \
		chunk=`printf "%04d" $${seed}`; \
		./generate --batch $(queue) --random-state $${seed} w.config --events $(events) --output $(output_path)/w_murfac20_$${chunk}.h5 --params "$(params);SigmaProcess:renormMultFac = 2.0"; \
	done
	for seed in $$(seq $(seed_offset) 1 $$(($(seed_offset) + 50))); do \
		chunk=`printf "%04d" $${seed}`; \
		./generate --batch $(queue) --random-state $${seed} qcd.config --events $(events) --output $(output_path)/qcd_murfac20_$${chunk}.h5 --params "$(params);SigmaProcess:renormMultFac = 2.0"; \
	done

pileup:
	DelphesPythia8 delphes_cards/delphes_converter_card.tcl pileup.config MinBias.root
	root2pileup MinBias.pileup MinBias.root
