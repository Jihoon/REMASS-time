import pymrio
test_mrio = pymrio.load_test()

print(test_mrio.get_sectors())
print(test_mrio.get_regions())

test_mrio.calc_all()