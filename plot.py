import sys
import numpy as np
from numpy import pi, r_
from scipy import optimize
import matplotlib.pyplot as plt

def main(argv):
    inputfile = sys.argv[1]
    #plt.rc('text', usetex = True)
    if "wavelength" in inputfile:
        wavelength(inputfile)
    if "lecher" in inputfile:
        lecher(inputfile)

def wavelength(inputfile):
    data = list()

    # get data from file
    with open(inputfile) as infile:
        next(infile)
        for line in infile:
            data.append([float(var) for var in line.strip().split("\t")])
    # sort data to according variable
    distance = np.array([point[0] for point in data])
    err_distance = np.array([point[1] for point in data])
    fieldstrength = np.array([point[2] for point in data])
    err_fieldstrength = np.array([point[4] for point in data])

    # fit data
    fitfunc = lambda p, x: p[0]*np.sin(2.*np.pi/p[1]*x+p[2])+p[3] # target function
    errfunc = lambda p, x, d: fitfunc(p, x) - d # disctance to target function
    p0 = [2.5,18.2,0.,5.]# initial guess
    p1, success = optimize.leastsq(errfunc, p0[:], args=(distance,fieldstrength)) # optimize

    # plot
    figure = plt.figure()
    distance_axis = np.linspace(distance.min(), distance.max(), 500)
    plt.errorbar(distance,fieldstrength,yerr=err_fieldstrength,xerr=err_distance,fmt="r.",capsize=2.)
    plt.plot(distance_axis, fitfunc(p1,distance_axis), "b-")
    plt.title("GUNN-Oszillator")
    plt.xlabel("Abstand d / mm")
    plt.ylabel("Spannung U / V")
    plt.ylim(1.,9.5)
    ax = plt.axes()
    legend = plt.legend(("Fit U(d)","Messwerte"), loc = 'upper right')
    plt.text(0.015, 0.93,
             'U(d) = %.2f * sin(2*pi*d/%.2f + %.2f) + %.2f' % (p1[0],p1[1],p1[2],p1[3]),
             fontsize=11,
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax.transAxes)
    plt.show()
    figure.savefig("GUNN.pdf", bbox_inches='tight')

def lecher(inputfile):
    data = list()

    # get data from file
    with open(inputfile) as infile:
        next(infile)
        for line in infile:
            data.append([float(var) for var in line.strip().split("\t")])
    # sort data to according variable
    distance = np.array([point[0] for point in data])
    fieldstrength = np.array([point[1] for point in data])
    err_fieldstrength = np.array([point[2] for point in data])

    # fit data
    fitfunc = lambda p, x: p[0]*np.sin(2*np.pi/p[1]*x+p[2]) + p[3]*np.exp(p[4]*x+p[5]) # target function
    #fitfunc = lambda p, x: p[0]*np.exp(p[1]*x+p[2])
    errfunc = lambda p, x, d: fitfunc(p, x) - d # disctance to target function
    p0 = [0.5,15,np.pi,2.,0.001,-0.52]# initial guess
    #p0 = [2.,0.001,-0.52]
    p1, success = optimize.leastsq(errfunc, p0[:], args=(distance,fieldstrength)) # optimize

    # plot
    figure = plt.figure()
    distance_axis = np.linspace(distance.min(), distance.max(), 1000)
    plt.errorbar(distance,fieldstrength,yerr=err_fieldstrength,fmt="r.",capsize=2.)
    plt.plot(distance_axis, fitfunc(p1,distance_axis), "b-")
    plt.title("Lecher-Leitung")
    plt.xlabel("Abstand d / mm")
    plt.ylabel("Spannung U / V")
    plt.ylim(1.,8.5)
    ax = plt.axes()
    legend = plt.legend(("Fit U(d)","Messwerte"),loc='lower right')
    plt.text(0.01, 0.93,
             'U(d) = %.2f*sin(2*pi/%.2f*d+%.2f) + %.2f*exp(%.2f*d+%.2f)' % (p1[0],p1[1],p1[2],p1[3],p1[4],p1[5]),
             #'U(d) = %.3f*exp(%.3f*d+%.3f)' % (p1[0],p1[1],p1[2]),
             fontsize=11,
             horizontalalignment='left',
             verticalalignment='top',
             transform=ax.transAxes)
    plt.show()
    figure.savefig("lecher.pdf", bbox_inches='tight')

if __name__ == "__main__":
    main(sys.argv)
