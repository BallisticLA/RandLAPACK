set terminal png size 1000,1000

#looping over m
do for [i in "4096 8192 16384"] {
    #looping n
    do for [j in "128 256 512"] {
        #Set in/out files
        #fname_in = sprintf("raw_data/test_%s_%s.dat", i, j)
        fname_in = sprintf("raw_data/test_QR_%s_%s.dat", i, j)

        # GEQR
        #reset
        stats fname_in u 4
        max = STATS_max
        min = STATS_min
        n=15 
        width_GEQR=(max-min)/n
        hist_GEQR(x,width_GEQR)=width_GEQR*floor(x/width_GEQR)+width_GEQR/2.0

        # CQR
        #reset
        stats fname_in u 1
        max = STATS_max
        min = STATS_min
        n=15 
        width_CQR=(max-min)/n
        hist_CQR(x,width_CQR)=width_CQR*floor(x/width_CQR)+width_CQR/2.0

        # HQR
        reset
        stats fname_in u 3
        max = STATS_max
        min = STATS_min
        n=15 
        width_HQR=(max-min)/n
        hist_HQR(x,width_HQR)=width_HQR*floor(x/width_HQR)+width_HQR/2.0
        
        # PLU
        #reset
        stats fname_in u 2
        max = STATS_max
        min = STATS_min
        n=15
        width_PLU=(max-min)/n
        hist_PLU(x,width_HQR)=width_PLU*floor(x/width_PLU)+width_PLU/2.0
        
        #set logscale #y
        #set logscale #x

        #set offset graph 0.05,0.05,0.05,0.0
        set boxwidth width_HQR*0.9
        set style fill solid 0.5 #fillstyle
        set tics out nomirror

        #fname_out = sprintf("hist_plots/test_%s_%s_COMBINED.png", i, j)
        fname_out = sprintf("hist_plots/test_QR_%s_%s_COMBINED.png", i, j)
        set output fname_out

        set xlabel "Runtime (μs)" #rotate by 180
        set ylabel "Frequency" #rotate by 270
        set title "Runtime Distribution" #rotate by 180

        
        #count and plot
        #plot fname_in u (hist_PLU($2,width_PLU)):(1.0) smooth freq w boxes lc rgb"blue" title 'PLUL', '' u (hist_CQR($1,width_CQR)):(1.0) smooth freq w boxes lc rgb"black" title 'CholQRQ', '' u (hist_HQR($3,width_HQR)):(1.0) smooth freq w boxes lc rgb"red" title 'HQRQ'#, '' u (hist_GEQR($4,width_GEQR)):(1.0) smooth freq w boxes lc rgb"purple" title 'GEQR'
        plot fname_in u (hist_HQR($3,width_HQR)):(1.0) smooth freq w boxes lc rgb"red" title 'HQRQ', '' u (hist_GEQR($4,width_GEQR)):(1.0) smooth freq w boxes lc rgb"purple" title 'GEQR'

    }
}