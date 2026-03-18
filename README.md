### INTRODUCTION
This repo, HY-DBSCAN also known as "HY(Hybrid) Density Based Clustering of Applications with Noise" is the implementation of the paper ["HY-DBSCAN: A hybrid parallel DBSCAN clustering algorithm scalable on distributed-memory computers"](https://doi.org/10.1016/j.jpdc.2022.06.005) by Guoqing Wu , Liqiang Cao, Hongyun Tian, Wei Wang.

### INSTALLATION OF DEPENDENCIES

1. mpjexpress
    - download the tar for mpjexpress from the following [website](https://sourceforge.net/projects/mpjexpress/files/releases/)
    - set the `MPJ_HOME` environment variable:- `export MPJ_HOME=/path/to/above/extracted/folder` (add it to .bashrc or .zshrc)

    NOTE:- you will have to resource your shell configurations for changes to take effect, if you are adding to your rc files.

### BUILDING THE PROJECT

- build the project using:- `mvn clean package`
- run the project by using:- `./run.sh` if you are using zsh, `./bashrun.sh` if you are using bash
- The third to last argument is the epsilon value, second to last minPts and last the input file
