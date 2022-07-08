set terminal png size 1000,1000
set logscale y

#looping over m
do for [i in "4096 8192 16384"] {
    #looping n
    do for [j in "128 256 512"] {
        #Set in/out files
        fname_in = sprintf("raw_data/test_%s_%s.dat", i, j)
        
        reset
        fname_out = sprintf("hist_plots/test_%s_%s_CQR.png", i, j)
        set output fname_out

        set ytics font ", 20"
        set xtics font ", 20"
        set key font ",20"
        set xtics offset 0, -1

        set lmargin 15
        set rmargin 6
        set ylabel offset -2,0
        set bmargin 8
        set xlabel offset 0,-2
        stats fname_in u 1

        max = STATS_max 
        min = STATS_min
        n=15
        width_CQR=(max-min)/n
        hist(x,width_CQR)=width_CQR*floor(x/width_CQR)+width_CQR/2.0
        
        set boxwidth width_CQR*0.9
        set style fill solid 0.5
        set tics out nomirror
        
        set title "{/*1.8 Runtime Distribution}"
        set xlabel "{/*1.8 Runtime (μs)}"
        set ylabel "{/*1.8 Frequency}"
        
        plot fname_in u (hist($1,width_CQR)):(1.0) smooth freq w boxes lc rgb"black" title 'CQR'
    
        reset
        fname_out = sprintf("hist_plots/test_%s_%s_PLU.png", i, j)
        set output fname_out

        set ytics font ", 20"
        set xtics font ", 20"
        set key font ",20"
        set xtics offset 0, -1

        set lmargin 15
        set rmargin 6
        set ylabel offset -2,0
        set bmargin 8
        set xlabel offset 0,-2
        stats fname_in u 2

        max = STATS_max 
        min = STATS_min
        n=15
        width_PLU=(max-min)/n
        hist(x,width_PLU)=width_PLU*floor(x/width_PLU)+width_PLU/2.0
        
        set boxwidth width_PLU*0.9
        set style fill solid 0.5
        set tics out nomirror
        
        set title "{/*1.8 Runtime Distribution}"
        set xlabel "{/*1.8 Runtime (μs)}"
        set ylabel "{/*1.8 Frequency}"
        
        plot fname_in u (hist($2,width_PLU)):(1.0) smooth freq w boxes lc rgb"blue" title 'PLU'
    
        reset
        fname_out = sprintf("hist_plots/test_%s_%s_HQR.png", i, j)
        set output fname_out
        set title 'Condition Numbers Distributions'    
        
        stats fname_in u 3
        set ytics font ", 20"
        set xtics font ", 20"
        set key font ",20"
        set xtics offset 0, -1

        set lmargin 15
        set rmargin 6
        set ylabel offset -2,0
        set bmargin 8
        set xlabel offset 0,-2

        max = STATS_max
        min = STATS_min
        n=15
        width_HQR=(max-min)/n 
        hist(x,width_HQR)=width_HQR*floor(x/width_HQR)+width_HQR/2.0
        
        set boxwidth width_HQR*0.9
        set style fill solid 0.5 #fillstyle
        set tics out nomirror
        
        set title "{/*1.8 Runtime Distribution}"
        set xlabel "{/*1.8 Runtime (μs)}"
        set ylabel "{/*1.8 Frequency}"
        
        plot fname_in u (hist($3,width_HQR)):(1.0) smooth freq w boxes lc rgb"red" title 'HQR'
        
        reset
        fname_out = sprintf("hist_plots/test_%s_%s_GEQR.png", i, j)
        set output fname_out
        set title 'Condition Numbers Distributions'

        set ytics font ", 20"
        set xtics font ", 20"
        set key font ",20"
        set xtics offset 0, -1

        set lmargin 15
        set rmargin 6
        set ylabel offset -2,0
        set bmargin 8
        set xlabel offset 0,-2
        stats fname_in u 4
        max = STATS_max #max value
        min = STATS_min #min value
        n=15 

        width_GEQR=(max-min)/n 
        hist(x,width_GEQR)=width_GEQR*floor(x/width_GEQR)+width_GEQR/2.0
            
    
        set boxwidth width_GEQR*0.9
        set style fill solid 0.5 #fillstyle
        set tics out nomirror
        
        set title "{/*1.8 Runtime Distribution}"
        set xlabel "{/*1.8 Runtime (μs)}"
        set ylabel "{/*1.8 Frequency}"
        
        plot fname_in u (hist($4,width_GEQR)):(1.0) smooth freq w boxes lc rgb"gold" title 'GEQR'
    }
}