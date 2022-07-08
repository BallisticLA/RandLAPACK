set terminal png size 1000,1000
set logscale y

#looping over m
do for [i in "4096 8192 16384"] {
    #looping n
    do for [j in "128 256 512"] {
        #Set in/out files
        fname_in = sprintf("raw_data/test_%s_%s.dat", i, j)
        fname_out = sprintf("hist_plots/test_%s_%s_CQR.png", i, j)

        set output fname_out
        reset
        set logscale x
        stats fname_in u 1
        max = STATS_max #max value
        min = STATS_min #min value
        n=15 #number of intervals
        width_CQR=(max-min)/n #interval width
        #function used to map a value to the intervals
        hist(x,width_CQR)=width_CQR*floor(x/width_CQR)+width_CQR/2.0
        set xrange [min:max]
        set yrange [0:]
        #to put an empty boundary around the
        #data inside an autoscaled graph.
        set offset graph 0.05,0.05,0.05,0.0
        set xtics min,(max-min)/5,max
        set boxwidth width_CQR*0.9
        set style fill solid 0.5 #fillstyle
        set tics out nomirror
        set title "{/*1.8 Runtime Distribution}"
        set xlabel "{/*1.8 Runtime (μs)}"
        set ylabel "{/*1.8 Frequency}"
        #count and plot
        plot fname_in u (hist($1,width_CQR)):(1.0) smooth freq w boxes lc rgb"black" title 'CQR'

        
        fname_out = sprintf("hist_plots/test_%s_%s_PLU.png", i, j)
        set output fname_out
        reset
        stats fname_in u 2
        max = STATS_max #max value
        min = STATS_min #min value
        n=15 #number of intervals
        width_PLU=(max-min)/n #interval width
        #function used to map a value to the intervals
        hist(x,width_HQR)=width_PLU*floor(x/width_PLU)+width_PLU/2.0
        set xrange [min:max]
        set yrange [0:]
        #to put an empty boundary around the
        #data inside an autoscaled graph.
        set offset graph 0.05,0.05,0.05,0.0
        set xtics min,(max-min)/5,max
        set boxwidth width_PLU*0.9
        set style fill solid 0.5 #fillstyle
        set tics out nomirror
        set title "{/*1.8 Runtime Distribution}"
        set xlabel "{/*1.8 Runtime (μs)}"
        set ylabel "{/*1.8 Frequency}"
        #count and plot
        plot fname_in u (hist($2,width_PLU)):(1.0) smooth freq w boxes lc rgb"blue" title 'PLU'
    

        fname_out = sprintf("hist_plots/test_%s_%s_HQR.png", i, j)
        set output fname_out
        set title 'Condition Numbers Distributions'
        reset
        stats fname_in u 3
        max = STATS_max #max value
        min = STATS_min #min value
        n=15 #number of intervals
        width_HQR=(max-min)/n #interval width
        #function used to map a value to the intervals
        hist(x,width_HQR)=width_HQR*floor(x/width_HQR)+width_HQR/2.0
        set xrange [min:max]
        set yrange [0:]
        #to put an empty boundary around the
        #data inside an autoscaled graph.
        set offset graph 0.05,0.05,0.05,0.0
        set xtics min,(max-min)/5,max
        set boxwidth width_HQR*0.9
        set style fill solid 0.5 #fillstyle
        set tics out nomirror
        set title "{/*1.8 Runtime Distribution}"
        set xlabel "{/*1.8 Runtime (μs)}"
        set ylabel "{/*1.8 Frequency}"
        #count and plot
        plot fname_in u (hist($3,width_HQR)):(1.0) smooth freq w boxes lc rgb"red" title 'HQR'
    
        fname_out = sprintf("hist_plots/test_%s_%s_GEQR.png", i, j)
        set output fname_out
        reset
        stats fname_in u 4
        max = STATS_max #max value
        min = STATS_min #min value
        n=15 #number of intervals
        width_GEQR=(max-min)/n #interval width
        #function used to map a value to the intervals
        hist(x,width_GEQR)=width_GEQR*floor(x/width_GEQR)+width_GEQR/2.0
        set xrange [min:max]
        set yrange [0:]
        #to put an empty boundary around the
        #data inside an autoscaled graph.
        set offset graph 0.05,0.05,0.05,0.0
        set xtics min,(max-min)/5,max
        set boxwidth width_GEQR*0.9
        set style fill solid 0.5 #fillstyle
        set tics out nomirror
        set title "{/*1.8 Runtime Distribution}"
        set xlabel "{/*1.8 Runtime (μs)}"
        set ylabel "{/*1.8 Frequency}"
        #count and plot
        plot fname_in u (hist($4,width_GEQR)):(1.0) smooth freq w boxes lc rgb"gold" title 'GEQR'
    }
}