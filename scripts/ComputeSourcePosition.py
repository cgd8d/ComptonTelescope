"""
Standalone compton telescope script.  It works by searching for the source position which optimizes the agreement with the expected compton scattering angles of events.

To use it, you will need the 3rd-party uncertainties module, available from https://pypi.python.org/pypi/uncertainties/.
"""

import sys
import math
import scipy.optimize
import uncertainties
import uncertainties.umath
import ROOT
ROOT.gSystem.Load("libEXOCalibUtilities")
if ROOT.gSystem.Load("libEXOCalibUtilities") < 0: sys.exit('Failed to load EXOCalibUtilities.')

# All positions are in mm.

# Size of bead -- according to Knut, they all have radii of roughly 0.35mm.
SourceSize = 0.35

# Position reconstruction accuracy comes from draft fitting note, April 2013.
UError = 2.4
VError = 1.2
ZError = 0.42

ElectronE = 510.998910
PrimaryE = None # Filled in based on source type from control records.  (Fix for MC.)

def UVtoXY(u, v, z):
    # Can't use EXOMiscUtil function because u and v are general objects.
    y = (u + v)/math.sqrt(3)
    if z > 0: x = v - u
    else: x = u - v
    return x, y

def Dot(v, w): return v[0]*w[0] + v[1]*w[1] + v[2]*w[2]

def GetAcceptableClusters(scint):
    # If the event has non-fiducial real clusters, skip by returning an empty array. 
    ret = []
    # We do not use the standard fiducial cuts.  This is because the telescope's sensitivity to
    # distance from the TPC is strongly dependent on its solid angle, as seen from the source;
    # so a stronger-than-necessary fiducial cut hurts our precision.
    # No systematic bias has been observed from cutting so close to the edges of the detector,
    # though in principle a position or energy bias at these locations could affect us.
    HexRad = 9.*19
    for i in range(scint.GetNumChargeClusters()):
        clu = scint.GetChargeClusterAt(i)
        if clu.fPurityCorrectedEnergy < 1: continue # Ignore clusters with no energy.
        if not clu.Is3DCluster(): return []
        if abs(clu.fU) > HexRad or abs(clu.fV) > HexRad or abs(clu.fX) > HexRad: return []
        if abs(clu.fZ) > 187: return []
        ret.append(clu)
    return ret

def GetResolution(energy):
    p0 = 9.71955e-01 ** 2
    p1 = 4.15237e+01 ** 2
    p2 = 2.33641e-02 ** 2
    return math.sqrt(p0*energy + p1 + p2*energy*energy)

def GetWeightedSum(x, y):
    # If x and y are two independent measurements of the same value,
    # then return a new value (also with uncertainties)
    # which weights them to get the best estimate of the underlying value,
    # and properly retains information about the correlated errors of the result.
    if x.std_dev == 0: return x
    if y.std_dev == 0: return y
    wx = 1./x.std_dev
    wy = 1./y.std_dev
    return (wx*x + wy*y)/(wx + wy)

class PrestoredCluster:
    def __init__(self, cluster):
        self.u = uncertainties.ufloat(cluster.fU, UError)
        self.v = uncertainties.ufloat(cluster.fV, VError)
        self.z = uncertainties.ufloat(cluster.fZ, ZError)
        self.x, self.y = UVtoXY(self.u, self.v, cluster.fZ)

class PrestoredEvent:
    def __init__(self, cluster1, cluster2, isMC):
        self.clu1 = PrestoredCluster(cluster1)
        self.clu2 = PrestoredCluster(cluster2)

        ######################################################################
        # Using the fact that the total energy is known, treat cluster1 and  #
        # cluster2 as two independent measurements of each other's energies. #
        ######################################################################

        # Start by grabbing the appropriate variable.
        # Not that we actually handle MC properly right now, but anyway...
        if isMC:
            e1 = cluster1.fCorrectedEnergy
            e2 = cluster2.fCorrectedEnergy
        else:
            e1 = cluster1.fPurityCorrectedEnergy
            e2 = cluster2.fPurityCorrectedEnergy

        # Produce values (with uncertainties) for each measurement taken independently.
        clu1E = uncertainties.ufloat(e1, GetResolution(e1))
        clu2E = uncertainties.ufloat(e2, GetResolution(e2))

        # Save the combination of the two results.
        # Weight them to give higher weight to the cluster with better energy resolution.
        # Note that self.clu1.E + self.clu2.E is guaranteed to be PrimaryE.
        self.clu1.E = GetWeightedSum(clu1E, PrimaryE - clu2E)
        self.clu2.E = GetWeightedSum(PrimaryE - clu1E, clu2E)

        # Pre-store some values we'll reuse many times.
        v12_temp = (self.clu2.x-self.clu1.x, self.clu2.y-self.clu1.y, self.clu2.z-self.clu1.z)
        dist = uncertainties.umath.sqrt(Dot(v12_temp, v12_temp))
        self.v12 = (v12_temp[0]/dist, v12_temp[1]/dist, v12_temp[2]/dist)

def GetChiSquare_ordered(xSource, ySource, zSource, clu1, clu2, event, sign):
    # Assuming clu1 happened before clu2, what is the chi-square of this scatter?

    # Angle of scatter.
    # Use precomputed normalized vector between clusters.
    v1 = (clu1.x-xSource, clu1.y-ySource, clu1.z-zSource)
    CosAngle = sign*Dot(v1, event.v12) / uncertainties.umath.sqrt(Dot(v1, v1))

    # Apply the compton scattering equation -- so, ideally this would equal zero.
    # This only works as-is for a two-cluster event, so clu2.E is the energy after the scatter.
    Result = 1. - CosAngle - ElectronE*(1./clu2.E - 1./PrimaryE)
    return pow(Result.nominal_value/Result.std_dev,2) # Sq-Sigmas away from 0.

def GetChiSquare(xSource, ySource, zSource, event):
    # Try both orderings.
    # We keep the best one -- the focus should be significantly improved when we do.
    # Note that this is a bad idea for anything other than source data.
    chi1 = GetChiSquare_ordered(xSource, ySource, zSource, event.clu1, event.clu2, event, 1.)
    chi2 = GetChiSquare_ordered(xSource, ySource, zSource, event.clu2, event.clu1, event, -1.)
    return min(chi1, chi2)

def GetTotalChiSquare(SourcePos, EventExtracts):
    chi2 = 0.
    xSource = uncertainties.ufloat(SourcePos[0], SourceSize)
    ySource = uncertainties.ufloat(SourcePos[1], SourceSize)
    zSource = uncertainties.ufloat(SourcePos[2], SourceSize)
    for event in EventExtracts:
        # If a single point disagrees at the 6-sigma level, it almost certainly
        # didn't originate at the source.  Could be Th background from elsewhere,
        # or from the source that was deflected without depositing much energy.
        # Preliminary studies show no benefit to cuts harsher than 6 sigma.
        chi2 += min(GetChiSquare(xSource, ySource, zSource, event), 6.**2)
    return chi2

def Run(prefix, **kwargs):

    # Only process source runs.
    beginRecord = kwargs['ControlRecordList'].GetNextRecord('EXOBeginRecord')()
    if not isinstance(beginRecord, ROOT.EXOBeginSourceCalibrationRunRecord):
        print "This is not a source run; skipping."
        return

    # Set the energy of the expected primary event.
    global PrimaryE
    if (beginRecord.GetSourceType() == ROOT.EXOBeginSourceCalibrationRunRecord.kThWeak or
        beginRecord.GetSourceType() == ROOT.EXOBeginSourceCalibrationRunRecord.kThStrong):
        PrimaryE = 2614.511
        SourceID = 'Th'
    elif (beginRecord.GetSourceType() == ROOT.EXOBeginSourceCalibrationRunRecord.kCoWeak or
          beginRecord.GetSourceType() == ROOT.EXOBeginSourceCalibrationRunRecord.kCoStrong):
        PrimaryE = 1332.492 # We only want single-gamma events; hopefully this is the easier one to select.
        SourceID = 'Co'
    elif (beginRecord.GetSourceType() == ROOT.EXOBeginSourceCalibrationRunRecord.kCsWeak or
          beginRecord.GetSourceType() == ROOT.EXOBeginSourceCalibrationRunRecord.kCsStrong):
        PrimaryE = 661.657
        SourceID = 'Cs'
    else:
        # We skip strong sources for now -- probably they could be handled fine,
        # but I don't have time to validate that right now.
        print "This run has source# " + str(beginRecord.GetSourceType())
        print "Must be an unrecognized one.  Returning..."
        return

    # Get the initial guess position for the source.
    xInit = 0.
    yInit = 0.
    zInit = 0.
    if beginRecord.GetSourcePosition() == ROOT.EXOBeginSourceCalibrationRunRecord.kP2_nz:
        zInit = -304.
        SourceID += '_P2nz'
    elif beginRecord.GetSourcePosition() == ROOT.EXOBeginSourceCalibrationRunRecord.kP2_pz:
        zInit = 304.
        SourceID += '_P2pz'
    elif beginRecord.GetSourcePosition() == ROOT.EXOBeginSourceCalibrationRunRecord.kP4_ny:
        yInit = -254.
        SourceID += '_P4ny'
    elif beginRecord.GetSourcePosition() == ROOT.EXOBeginSourceCalibrationRunRecord.kP4_py:
        yInit = 254.
        SourceID += '_P4py'
    elif beginRecord.GetSourcePosition() == ROOT.EXOBeginSourceCalibrationRunRecord.kP4_px:
        xInit = 254.
        SourceID += '_P4px'
    else:
        print "Unrecognized source position -- er, I think most of these are near S5.  Let's try it."
        xInit = 254.
        SourceID += '_other'

    EventChain = kwargs['EventTree']
    event = ROOT.EXOEventData()
    EventChain.SetBranchAddress("EventBranch", event)

    # Calib objects we'll reuse.
    calibManager = ROOT.EXOCalibManager.GetCalibManager()
    eCalib = ROOT.EXOEnergyCalib.GetInstanceForFlavor("2013-0nu-denoised","2013-0nu-denoised","vanilla")

    # Locate the entries worth scanning each time, so we only have to waste time once.
    EventExtracts = []
    for i in xrange(EventChain.GetEntries()):
        EventChain.GetEntry(i)

        for j in xrange(event.GetNumScintillationClusters()):
            scint = event.GetScintillationCluster(j)

            # Require two clusters with positive energy.
            # Beyond two clusters, it becomes harder to order them properly.  To study.
            clusters = GetAcceptableClusters(scint)
            if len(clusters) != 2: continue

            # Cut on anticorrelated energy.
            # Note that we'll need the individual cluster energies (charge-only),
            # But using anticorrelated energy here improves the effectiveness of our cut on the full-energy peak.
            scintE = scint.fDenoisedEnergy
            # We currently apply a correction to the scintillation energy.
            sum_charge_e = sum(clu.fPurityCorrectedEnergy for clu in clusters)
            sum_correction = 0.
            for clu in clusters:
                if clu.fZ > 0:
                    corr = (0.9355 + 1.289*pow(abs(clu.fZ/1e3), 2.004))
                else:
                    corr = (0.938 + 0.6892*pow(abs(clu.fZ/1e3), 1.716))
                sum_correction += corr*clu.fPurityCorrectedEnergy
            scintE /= sum_correction/sum_charge_e
            # Now apply the denoised-scintillation calibration.
            antiCorrE = eCalib.CalibratedRotatedEnergy(sum(clu.fPurityCorrectedEnergy for clu in clusters),
                                                       scint.fRawEnergy,
                                                       2,
                                                       event.fEventHeader)

            # Is it consistent with 2615 keV?  If so, add it to the TEventList.
            # Consistent here will mean within 1 sigma of 2615 keV.
            EResCalib = calibManager.getCalib("energy-resolution",
                                              "2013-0nu-denoised-weekly",
                                              event.fEventHeader)
            resolution = EResCalib.RotatedResolution(PrimaryE, 2)
            if abs(PrimaryE - antiCorrE) < resolution:
                EventExtracts.append(PrestoredEvent(clusters[0], clusters[1], False))

    print "Done selecting events; selected",len(EventExtracts)
    if len(EventExtracts) < 20:
        print "Fewer than 20 events passed the cuts; skipping this run."
        return

    # import antigravity
    result = scipy.optimize.fmin_bfgs(GetTotalChiSquare,
                                     [xInit, yInit, zInit],
                                     args = (EventExtracts,),
                                     epsilon = 1.e-3, # default leads to indistinguishable changes in chi2.
                                     gtol = 1.e-1, # default is 1.e-5, but we don't need that accuracy.
                                     full_output = True)
    print "return value from optimization is ", result

    retDict = {}
    retDict[SourceID + '_x'] = (result[0][0], result[3][0][0])
    retDict[SourceID + '_y'] = (result[0][1], result[3][1][1])
    retDict[SourceID + '_z'] = (result[0][2], result[3][2][2])
    return retDict

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'We need the run number to be passed in as an argument:'
        print 'python ComputeRotationAngle.py <run_number>'
        sys.exit()
    ROOT.gROOT.SetBatch()
    ROOT.gSystem.Load("libEXOUtilities")
    if ROOT.gSystem.Load("libEXOUtilities") < 0: sys.exit('Failed to load EXOUtilities.')
    import glob
    try:
        # If an integer was passed in, interpret it as a run number (at SLAC).
        globname = '/nfs/slac/g/exo_data3/exo_data/data/WIPP/DN_Source_LJPurity_Jan2014/' + str(int(sys.argv[1])) + '/denoised*.root'
    except ValueError:
        # If we couldn't convert it from an integer, interpret it as a filename.
        globname = sys.argv[1]
    print globname
    EventTree = ROOT.TChain('tree')
    EventTree.Add(globname)
    AllMaskedFiles = glob.glob(globname)
    AllMaskedFiles.sort(reverse = True)
    LastFile = ROOT.TFile(AllMaskedFiles[0])
    LastTree = LastFile.Get('tree')
    ControlRecordList = LastTree.GetUserInfo().At(1)
    result = Run("ComptonScript", EventTree = EventTree, ControlRecordList = ControlRecordList)
    print result


