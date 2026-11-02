# Build and run every test of one version:  make cpu_batch  /  make cpu_inline  /  make cuda
.PHONY: cpu_batch cpu_inline cuda clean

cpu_batch:
	$(MAKE) -C cpu/cpu_batch run

cpu_inline:
	$(MAKE) -C cpu/cpu_inline run

cuda:
	$(MAKE) -C cuda run

clean:
	$(MAKE) -C cpu/cpu_batch clean
	$(MAKE) -C cpu/cpu_inline clean
	$(MAKE) -C cuda clean
