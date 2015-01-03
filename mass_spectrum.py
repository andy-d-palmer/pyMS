class mass_spectrum():
    # a class that describes a single mass spectrum
    def __init__(self):
        self.mzs = []
        self.intensities = []
        self.centroids = [] 
    def add_intensities(self,intensities):
        self.intensities = intensities
    def add_mzs(self,mzs):
        self.mzs = mzs




