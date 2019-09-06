import glob
import numpy as np

tolerance = 0.0001
spring_constant = 1.0
temperature = 1.0
beta = 1.0 / (1.0 * temperature)

N_BB = (50, 100, 200)
Point = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0)

num_windows = len(Point)

for N in N_BB[2:]:
    for iter in range(1, 10):
        num_bins = 0
        filenames = glob.glob('L-' + str.format('%d' % N) + '-' + str.format('%.0f' % Point[0]) + '_' + str.format(
            '%d' % iter) + '.histo')
        with open(filenames[0], 'r') as f:
            lines = f.readlines()
            num_bins = len(lines) - 4
            min = float(lines[4].split()[1])
            max = float(lines[-1].split()[1])
            f.close()
        bin_width = round((max - min) / (num_bins - 1), 1)
        restraints = np.zeros(num_windows)
        histo_minmax = np.zeros((num_windows, 2))
        old_histogram = np.zeros((num_windows, num_bins))
        histototal = np.zeros(num_windows)
        F_old = np.zeros((num_windows))
        F_new = np.zeros((num_windows))
        for index_window, i_Point in enumerate(Point):
            # read data
            filename = 'L-' + str.format('%d' % N) + '-' + str.format('%.0f' % i_Point) + '_' + str.format(
                '%d' % iter) + '.histo'
            f = open(filename, 'r')
            lines = f.readlines()
            restraints[index_window] = float(filename.replace('.histo', '').split('_')[-2].split('-')[-1])
            for i, line_i in enumerate(lines[4:]):
                num, binvalue, count, fraction = map(float, line_i.split())
                if i == 0:
                    histo_minmax[index_window][0] = binvalue
                elif i == num_bins - 1:
                    histo_minmax[index_window][1] = binvalue
                old_histogram[index_window][i] = count
                histototal[index_window] += count
            f.close()

        hi_boundary = np.max(histo_minmax[:, 1])
        lo_boundary = np.min(histo_minmax[:, 0])

        single_window_num_bins = num_bins
        num_bins = int(round((hi_boundary - lo_boundary) / bin_width)) + 1
        histogram = np.zeros((num_windows, num_bins))
        Probability = np.zeros((num_bins))
        Free_Energy = np.zeros(num_bins)
        for i in range(num_windows):
            startpoint = int(round((histo_minmax[i][0] - lo_boundary) / bin_width))
            endpoint = startpoint + single_window_num_bins
            histogram[i][startpoint:endpoint] = np.copy(old_histogram[i])

        # iteration
        error = 1.0
        while error > tolerance:
            for i in range(num_bins):
                num = 0
                denom = 0
                coor = lo_boundary + i * bin_width - bin_width / 2.0
                for index_window in range(num_windows):
                    num += histogram[index_window][i]
                    potential_bias = 0.5 * spring_constant * (coor - restraints[index_window]) ** 2.0
                    denom += np.exp((F_old[index_window] - potential_bias) * beta) * histototal[index_window]
                Probability[i] = num / denom

                for index_window in range(num_windows):
                    potential_bias = 0.5 * spring_constant * (coor - restraints[index_window]) ** 2.0
                    F_new[index_window] += np.exp(-potential_bias * beta) * Probability[i]

            for i in range(0, num_windows):
                F_new[i] = (-1 / beta) * np.log(F_new[i])

            difference = F_new[0]
            F_new[:] -= difference

            error = np.mean(np.abs(F_new - F_old))
            F_old = np.copy(F_new)
            F_new[:] = 0

        Free_Energy[:] = -(1 / beta) * np.log(Probability[:])
        min = np.min(Free_Energy)
        Free_Energy[:] -= min

        g = open('L-' + str.format('%d' % N) + '-' + str.format('%d' % iter) + '.csv', 'w')
        for i in range(num_bins):
            if Probability[i] != 0:
                coor = lo_boundary + i * bin_width - bin_width / 2.0
                g.write(str('%.2f' % round(coor, 2)) + ', ' + str('%.4f' % Free_Energy[i]) + '\n')
        g.close()



