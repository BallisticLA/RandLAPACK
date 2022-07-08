
set terminal png size 1000,1000
set logscale y

#Set line styles
set style line 1 \
linecolor rgb '#0060ad' \
linetype 1 linewidth 1.5 \
pointtype 7 pointsize 1.0

#looping over k
do for [i in "1024 2048"] {
    #looping over block size
    do for [j in "64 128 256"] {
        #looping over power iters
        do for [k in "0 2"] {
            # looping over decay {
            do for [dec in "0 2"] {

                # PLOTTING RF
                #Set in/out files
                fname_in = sprintf("raw_data/test_RF_%s_%s_%s_%s.dat", i, j, k, dec)
                fname_out = sprintf("plots/test_RF_%s_%s_%s_%s.png", i, j, k, dec)
                set output fname_out
                set title '{/*1.8 Condition Numbers of Sketches}'
                set ylabel "{/*1.8 Condition Number Value}"
                set xlabel "{/*1.8 QB Iteration}"

                plot fname_in u 0:2:xtic(8) with linespoints linestyle 1 notitle, '' u 0:3:xtic(8) with linespoints linestyle 1 notitle, '' u 0:4:xtic(8) with linespoints linestyle 1 notitle, '' u 0:5:xtic(8) with linespoints linestyle 1 notitle, '' u 0:6:xtic(8) with linespoints linestyle 1 notitle
            
                if (k != 0) {
                    # PLOTTING RS
                    #Set in/out files
                    fname_in = sprintf("raw_data/test_RS_%s_%s_%s_%s.dat", i, j, k, dec)
                    fname_out = sprintf("plots/test_RS_%s_%s_%s_%s.png", i, j, k, dec)
                    set output fname_out
                    set title '{/*1.8 Condition Numbers}'
                    set ylabel "{/*1.8 Condition Number Value}"
                    set xlabel "{/*1.8 Pass Over Data}"

                    plot fname_in u 0:2:xtic(8) with linespoints linestyle 1 notitle, '' u 0:3:xtic(8) with linespoints linestyle 1 notitle, '' u 0:4:xtic(8) with linespoints linestyle 1 notitle, '' u 0:5:xtic(8) with linespoints linestyle 1 notitle, '' u 0:6:xtic(8) with linespoints linestyle 1 notitle            
                }
            }
        }
    }
}