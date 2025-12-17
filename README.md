## Install Spack for YAML-Cpp

- Install Spack to system:

```bash
cd ~
git clone -c feature.manyFiles=true https://github.com/spack/spack.git
```

- Install YAML-Cpp using Spack

```bash
source ~/spack/share/spack/setup-env.sh
spack list yaml
spack install yaml-cpp
```