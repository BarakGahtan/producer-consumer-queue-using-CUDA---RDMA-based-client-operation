///////////////////////////////////////////////// DO NOT CHANGE ///////////////////////////////////////

#include "ex3.h"

void cpu_process(uchar *img_in, uchar *img_out, int width, int height) {
    int histogram[256] = { 0 };
    for (int i = 0; i < width * height; i++) {
        histogram[img_in[i]]++;
    }

    int cdf[256] = { 0 };
    int hist_sum = 0;
    for (int i = 0; i < 256; i++) {
        hist_sum += histogram[i];
        cdf[i] = hist_sum;
    }

    uchar map[256] = { 0 };
    for (int i = 0; i < 256; i++) {
        float map_value = float(cdf[i]) / (width * height);
        map[i] = ((uchar)(N_COLORS * map_value)) * (256 / N_COLORS);
    }

    for (int i = 0; i < width * height; i++) {
        img_out[i] = map[img_in[i]];
    }
}
