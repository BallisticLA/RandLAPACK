
set terminal png size 1000,1000
set logscale y
set ytics font ", 20"
set xtics font ", 20"
set key font ",20"
set xtics offset 0, -1

set lmargin 15
set ylabel offset -2,0
set bmargin 8
set xlabel offset 0,-2

#Set line styles
set style line 1 \
linecolor rgb '#0060ad' \
linetype 1 linewidth 1.5 \
pointtype 7 pointsize 1.0

#Set line styles
set style line 2 \
linecolor rgb '#ff0000' \
linetype 1 linewidth 1.5 \
pointtype 7 pointsize 1.0

#looping over k
do for [i in "1024 2048"] {
    #looping over block size
    do for [j in "64 128 256"] {
        #looping over power iters
        do for [k in "0 2"] {

            # PLOTTING RF
            #Set in/out files
            fname_in_0 = sprintf("raw_data/test_RF_%s_%s_%s_0.dat", i, j, k)
            fname_in_2 = sprintf("raw_data/test_RF_%s_%s_%s_2.dat", i, j, k)

            fname_out = sprintf("plots/test_RF_%s_%s_%s.png", i, j, k)
            set output fname_out
            set title '{/*1.8 Condition Numbers of Sketches}'
            set ylabel "{/*1.8 Condition Number Value}"
            set xlabel "{/*1.8 QB Iteration}"

            plot fname_in_0 u 0:2:xtic(8) with linespoints linestyle 1 title "Fast Decay", '' u 0:3:xtic(8) with linespoints linestyle 1 notitle, '' u 0:4:xtic(8) with linespoints linestyle 1 notitle, '' u 0:5:xtic(8) with linespoints linestyle 1 notitle, '' u 0:6:xtic(8) with linespoints linestyle 1 notitle, \
                    fname_in_2 u 0:2:xtic(8) with linespoints linestyle 2 title "Slow Decay", '' u 0:3:xtic(8) with linespoints linestyle 2 notitle, '' u 0:4:xtic(8) with linespoints linestyle 2 notitle, '' u 0:5:xtic(8) with linespoints linestyle 2 notitle, '' u 0:6:xtic(8) with linespoints linestyle 2 notitle

            if (k != 0) {
                # PLOTTING RS
                #Set in/out files
                fname_in_0 = sprintf("raw_data/test_RS_%s_%s_%s_0.dat", i, j, k)
                fname_in_2 = sprintf("raw_data/test_RS_%s_%s_%s_2.dat", i, j, k)

                fname_out = sprintf("plots/test_RS_%s_%s_%s.png", i, j, k)
                set output fname_out
                set title '{/*1.8 Condition Numbers}'
                set ylabel "{/*1.8 Condition Number Value}"
                set xlabel "{/*1.8 Pass Over Data}"

                plot fname_in_0 u 0:2:xtic(8) with linespoints linestyle 1 title "Fast Decay", '' u 0:3:xtic(8) with linespoints linestyle 1 notitle, '' u 0:4:xtic(8) with linespoints linestyle 1 notitle, '' u 0:5:xtic(8) with linespoints linestyle 1 notitle, '' u 0:6:xtic(8) with linespoints linestyle 1 notitle, \
                        fname_in_2 u 0:2:xtic(8) with linespoints linestyle 2 title "Slow Decay", '' u 0:3:xtic(8) with linespoints linestyle 2 notitle, '' u 0:4:xtic(8) with linespoints linestyle 2 notitle, '' u 0:5:xtic(8) with linespoints linestyle 2 notitle, '' u 0:6:xtic(8) with linespoints linestyle 2 notitle 
            }
        }
    }
}