export OMP_NUM_THREADS=1
export LOGFILENAME="run_omp_num_threads_${OMP_NUM_THREADS}_1t_log.txt"
export RUBYDIR="/home/rjmurr/laps/RandLAPACK/demos/sparse-data-matrices/ruby-graphs/"

export GRAPHFILE="${RUBYDIR}/HU_edges/HU_edges (47538, 9.37).mtx"
export LOGFILE="${RUBYDIR}/HU_edges/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"

export GRAPHFILE="${RUBYDIR}/soc-Epinions1/soc-Epinions1 (75888, 5.34).mtx"
export LOGFILE="${RUBYDIR}/soc-Epinions1/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"

export GRAPHFILE="${RUBYDIR}/musae_facebook_edges/musae_facebook_edges (22470, 7.61).mtx"
export LOGFILE="${RUBYDIR}/musae_facebook_edges/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"


export OMP_NUM_THREADS=2
export LOGFILENAME="run_omp_num_threads_${OMP_NUM_THREADS}_1t_log.txt"
export RUBYDIR="/home/rjmurr/laps/RandLAPACK/demos/sparse-data-matrices/ruby-graphs/"

export GRAPHFILE="${RUBYDIR}/HU_edges/HU_edges (47538, 9.37).mtx"
export LOGFILE="${RUBYDIR}/HU_edges/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"

export GRAPHFILE="${RUBYDIR}/soc-Epinions1/soc-Epinions1 (75888, 5.34).mtx"
export LOGFILE="${RUBYDIR}/soc-Epinions1/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"

export GRAPHFILE="${RUBYDIR}/musae_facebook_edges/musae_facebook_edges (22470, 7.61).mtx"
export LOGFILE="${RUBYDIR}/musae_facebook_edges/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"


export OMP_NUM_THREADS=3
export LOGFILENAME="run_omp_num_threads_${OMP_NUM_THREADS}_1t_log.txt"
export RUBYDIR="/home/rjmurr/laps/RandLAPACK/demos/sparse-data-matrices/ruby-graphs/"

export GRAPHFILE="${RUBYDIR}/HU_edges/HU_edges (47538, 9.37).mtx"
export LOGFILE="${RUBYDIR}/HU_edges/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"

export GRAPHFILE="${RUBYDIR}/soc-Epinions1/soc-Epinions1 (75888, 5.34).mtx"
export LOGFILE="${RUBYDIR}/soc-Epinions1/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"

export GRAPHFILE="${RUBYDIR}/musae_facebook_edges/musae_facebook_edges (22470, 7.61).mtx"
export LOGFILE="${RUBYDIR}/musae_facebook_edges/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"


export OMP_NUM_THREADS=4
export LOGFILENAME="run_omp_num_threads_${OMP_NUM_THREADS}_1t_log.txt"
export RUBYDIR="/home/rjmurr/laps/RandLAPACK/demos/sparse-data-matrices/ruby-graphs/"

export GRAPHFILE="${RUBYDIR}/HU_edges/HU_edges (47538, 9.37).mtx"
export LOGFILE="${RUBYDIR}/HU_edges/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"

export GRAPHFILE="${RUBYDIR}/soc-Epinions1/soc-Epinions1 (75888, 5.34).mtx"
export LOGFILE="${RUBYDIR}/soc-Epinions1/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"

export GRAPHFILE="${RUBYDIR}/musae_facebook_edges/musae_facebook_edges (22470, 7.61).mtx"
export LOGFILE="${RUBYDIR}/musae_facebook_edges/${LOGFILENAME}"
./rchol_shift_inv 1 "$GRAPHFILE" | tee -a "$LOGFILE"
