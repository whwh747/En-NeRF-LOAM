**add FF encodeing for NeRF-LOAM**

note: if you want to run maicity00 sequence, you should decrease [there](src/variations/render_helpers.py#L264), for example 
```chunk_size//20```, this will significantly increase the runtime!