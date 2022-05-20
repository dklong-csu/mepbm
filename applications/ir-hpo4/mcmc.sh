n_runs=2
first_seed=4
last_seed=`expr $first_seed + $n_runs - 1`
for i in `seq $first_seed $last_seed`
do
	cp mcmc.m mcmc${i}.m
	sed -i "s/seed = 1/seed = ${i}/g" mcmc${i}.m
	nohup nice -19 matlab -batch "mcmc${i}; exit" > nohup${i}.out &
done