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
linecolor rgb '#000000' \
linetype 1 linewidth 2.0\
pointtype 7 pointsize 1.0

#Set line styles
set style line 2 \
linecolor rgb '#0080ff' \
linetype 1 linewidth 2.0 \
pointtype 7 pointsize 1.0

#Set line styles
set style line 3 \
linecolor rgb '#ff0000' \
linetype 1 linewidth 2.0 \
pointtype 7 pointsize 1.0

#Set line styles
set style line 4 \
linecolor rgb '#8A2BE2' \
linetype 1 linewidth 2.0 \
pointtype 7 pointsize 1.0

#looping over m
do for [i in "4096 8192 16384"] {
    #looping n
        #Set in/out files
        fname_in = sprintf("raw_data/test_mean_time_%s.dat", i)
        #fname_in = sprintf("raw_data/test__mean_time_QR_%s.dat", i)
        fname_out = sprintf("plots/test_mean_time_%s.png", i)
        #fname_out = sprintf("plots/test_mean_time_QR_%s.png", i)
        set output fname_out
        set title "{/*1.8 Stabilization Speed Tests}"
        set xlabel "{/*1.8 Column size}"
        set ylabel "{/*1.8 Runtime (μs)}"

        set xtics ("" 0, "128" 1, "" 2, "256" 3, "" 4, "384" 5, "" 6, "512" 7)

        # GEQR Commented out
        plot fname_in u 0:1:xtic(8) with linespoints linestyle 1 title "CholQRQ", '' u 0:2:xtic(8) with linespoints linestyle 2 title "PLUL", '' u 0:3:xtic(8) with linespoints linestyle 3 title "HQRQ"#,  '' u 0:4:xtic(8) with linespoints linestyle 4 title "GEQR"
        
        # Check only HQR vs GEQR
        #plot fname_in u 0:3:xtic(8) with linespoints linestyle 3 title "HQRQ",  '' u 0:4:xtic(8) with linespoints linestyle 4 title "GEQR"
}