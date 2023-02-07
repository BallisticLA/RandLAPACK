
        if(this->verbosity) {
            switch(termination) {
            case 1:
                printf("\nCholQRCP TERMINATED VIA: 1.\n");
                break;
            case 0:
                printf("\nCholQRCP TERMINATED VIA: normal termination.\n");
                break;
            }
        }
        return termination;
    }
};
#endif
} // end namespace RandLAPACK::comps::cholqrcp