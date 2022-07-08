set terminal png size 1000,1000
# REMEMBER THAT FIRST COLUMN IS COUNTER

#looping over k
do for [i in "1024 2048"] {
    #looping over block size
    do for [j in "64 128 256"] {
        #looping over power iters
        do for [k in "0 2"] {
            # looping over decay {
            do for [dec in "0 2"] {
                #Set in/out files
                fname_in = sprintf("raw_data/test_%s_%s_%s_%s.dat", i, j, k, dec)
                n = 20000

                # Column 2
                #reset
                stats fname_in u 2
                max = STATS_max
                min = STATS_min
                width_2=(max-min)/n
                hist_2(x,width_2)=width_2*floor(x/width_2)+width_2/2.0

                # Column 3
                stats fname_in u 3
                max = STATS_max
                min = STATS_min
                width_3=(max-min)/n
                hist_3(x,width_3)=width_3*floor(x/width_3)+width_3/2.0
                
                # Column 4
                stats fname_in u 4
                max = STATS_max
                min = STATS_min 
                width_4=(max-min)/n
                hist_4(x,width_4)=width_4*floor(x/width_4)+width_4/2.0

                # Column 5
                stats fname_in u 5
                max = STATS_max
                min = STATS_min
                width_5=(max-min)/n
                hist_5(x,width_5)=width_5*floor(x/width_5)+width_5/2.0

                #set logscale y
                #set logscale x

                set offset graph 0.05,0.05,0.05,0.0
                set boxwidth width_2*1
                set style fill solid 0.5 #fillstyle

                fname_out = sprintf("hist_plots/test_%s_%s_%s_%s.png", i, j, k, dec)
                set output fname_out

                set xlabel "Condition Number Value"
                set ylabel "Frequency"
                set title "Condition Number Distribution"

                
                #count and plot
                plot fname_in u (hist_2($2,width_2)):(1.0) smooth freq w boxes lc rgb"blue" notitle, '' u (hist_3($3,width_3)):(1.0) smooth freq w boxes lc rgb"blue" notitle, '' u (hist_4($4,width_4)):(1.0) smooth freq w boxes lc rgb"blue" notitle, '' u (hist_5($5,width_5)):(1.0) smooth freq w boxes lc rgb"blue" notitle
            }
        }
    }
}