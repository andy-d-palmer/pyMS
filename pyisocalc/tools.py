def make_sf_adduct_database(sum_formulae,adducts,output_filename,sigma=0.001,resolution=10000,charge=1):
    from pyMS.pyisocalc import pyisocalc
    # Extract variables from config dict
    # Check if already genrated and load if possible, otherwise calculate fresh
    with open(output_filename,'a') as f_out:
        for sum_formula in sum_formulae:
            #print sum_formula
            for adduct in adducts:
                try:
                    sf = pyisocalc.complex_to_simple(sum_formula+adduct)
                    if sf is None: # not possible to form adduct
                        continue

                    isotope_ms = pyisocalc.isodist(sf, plot=False, sigma=sigma, charges=charge,
                                               resolution=resolution)
                except KeyError as e:
                    if str(e).startswith("KeyError:"):
                        print str(e)
                        continue
                except ValueError as e:
                    if str(e).startswith("Element not recognised"):
                        print str(e)
                        continue
                except:
                    print sf=="", sum_formula, adduct
                    raise
                f_out.write("{},[M{}],{},{}\n".format(sum_formula,adduct,isotope_ms.get_spectrum(source='centroids')[0],isotope_ms.get_spectrum(source='centroids')[1]))


def make_sf_adduct_optimusfilter(sum_formulae,adducts,output_filename,sigma=0.001,resolution=10000,charge=1):
    from pyMS.pyisocalc import pyisocalc
    # Extract variables from config dict
    # Check if already genrated and load if possible, otherwise calculate fresh
    with open(output_filename,'a') as f_out:
        for sum_formula in sum_formulae:
            #print sum_formula
            for adduct in adducts:
                try:
                    sf = pyisocalc.complex_to_simple(sum_formula+adduct)
                    if sf is None: # not possible to form adduct
                        continue

                    isotope_ms = pyisocalc.isodist(sf, plot=False, sigma=sigma, charges=charge,
                                               resolution=resolution)
                except KeyError as e:
                    if str(e).startswith("KeyError:"):
                        print str(e)
                        continue
                except ValueError as e:
                    if str(e).startswith("Element not recognised"):
                        print str(e)
                        continue
                except:
                    print sf=="", sum_formula, adduct
                    raise
                f_out.write("{} [M{}],-1,{}\n".format(sum_formula,adduct,isotope_ms.get_spectrum(source='centroids')[0][0]))


