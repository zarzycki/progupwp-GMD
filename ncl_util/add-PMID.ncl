begin

f = addfile(fname,"w")

print(fname)

PS = f->PS
hyam = f->hyam
hybm = f->hybm

P0=100000.
PMID = pres_hybrid_ccm_se(PS,P0,hyam,hybm)

f->PMID = PMID

exit

end