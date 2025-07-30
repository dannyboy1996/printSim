\# PrintSim



\*\*PrintSim\*\* is a command-line tool that takes standard G-code (Marlin-compatible; other flavors not yet tested) and converts it into a `.wav` audio file. The result simulates the sounds of a real 3D printer, allowing you to \*hear\* your print before you even start it.



---



\### Features



\* Accurate stepper motor sound modeling

\* Simulated fan sounds:



  \* Mainboard fan

  \* PSU fan

  \* Hotend fan

  \* Part-cooling fan

\* Print bed resonance modeling for realistic depth and texture



---



\### 🚀 Usage



```cmd

python printsim.py your\\\_gcode\\\_file.gcode

```



This will generate a `.wav` file simulating your print.



---



\### 🛠️ To-Do



\* Support for arc move sounds (`G2` and `G3`)

\* More accurate emulation of part-cooling fan spin-up/spin-down behavior, like on real printers

Please feel free to submit a pull-request with more features



