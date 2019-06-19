#!/usr/bin/env python3

# #############################################################################
# build.py
# ========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Pypeline build script.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request


def parse_args():
    parser = argparse.ArgumentParser(
        description="Install Pypeline tools.",
        epilog="""
            Examples
            --------
            python3 build.py --doc
            python3 build.py --lib=Debug
                             --C_compiler /usr/bin/clang
                             --CXX_compiler /usr/bin/clang++
            python3 build.py --install_dependencies
                             --C_compiler /usr/bin/gcc-7
                             --CXX_compiler /usr/bin/gcc-7
                                            """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--lib",
        help="Compile C++/Python libraries in Debug or Release mode.",
        type=str,
        choices=["Debug", "Release"],
    )
    group.add_argument("--doc", help="Generate HTML documentation.", action="store_true")
    group.add_argument(
        "--download_dependencies",
        help="Download dependencies and extract archives.",
        action="store_true",
    )
    group.add_argument(
        "--install_dependencies", help="Install Pypeline's C++ dependencies.", action="store_true"
    )
    parser.add_argument(
        "--C_compiler",
        help="C compiler executable. Use system default if unspecified.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--CXX_compiler",
        help="C++ compiler executable. Use system default if unspecified.",
        type=str,
        required=False,
    )
    parser.add_argument("--OpenMP", help="Use OpenMP", action="store_true")
    parser.add_argument(
        "--print",
        help="Only print commands that would have been executed given specified options.",
        action="store_true",
    )

    args = parser.parse_args()
    return args


def build_lib(args, project_root_dir):
    build_dir = f"{project_root_dir}/build/pypeline"
    cmds = f"""
        source "{project_root_dir}/pypeline.sh" --no_shell;
        rm -rf "{build_dir}";
        mkdir --parents "{build_dir}";
        cd "{build_dir}";
        cmake -DCMAKE_BUILD_TYPE="{args.lib}" \
        {f'-DCMAKE_C_COMPILER="{args.C_compiler}"' if (args.C_compiler is not None) else ''} \
        {f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}"' if (args.CXX_compiler is not None) else ''} \
           -DPYPELINE_USE_OPENMP={str(args.OpenMP).upper()} \
           "{project_root_dir}";
        make install;
        cd "{project_root_dir}";
        # python3 "{project_root_dir}/setup.py" develop;
        """
    return cmds


def build_doc(args, project_root_dir):
    cmds = f"""
        source "{project_root_dir}/pypeline.sh" --no_shell;
        python3 "{project_root_dir}/setup.py" build_sphinx;
        """
    return cmds


def download_dependencies(args, project_root_dir):
    archive_dir = project_root_dir / "dependencies"

    if not archive_dir.exists():
        archive_dir.mkdir(parents=True)

    for web_link in [
        "http://bitbucket.org/eigen/eigen/get/3.3.7.tar.gz",
        "https://github.com/pybind/pybind11/archive/v2.3.0.tar.gz",
        "http://www.fftw.org/fftw-3.3.8.tar.gz",
    ]:
        print(f"Downloading {web_link}")
        with urllib.request.urlopen(web_link) as response:
            archive_path = archive_dir / os.path.basename(web_link)
            with archive_path.open(mode="wb") as archive:
                shutil.copyfileobj(response, archive)

        with tarfile.open(archive_path) as archive:
            extracted_dir = archive_dir / os.path.commonprefix(archive.getnames())
            archive.extractall(path=extracted_dir)

    cmds = ""  # for '--print' to not fail if specified.
    return cmds


def find_dependency(name, project_root_dir):
    """
    Given the name of a dependency, find the directory in dependencies/ to which it was extracted.
    """
    dep_dir = project_root_dir / "dependencies"
    candidates = [_ for _ in dep_dir.iterdir() if _.is_dir() and (name in _.name)]
    if len(candidates) == 1:
        return candidates[0].absolute() / candidates[0].name
    else:
        raise ValueError(f'Could not locate directory containing "{name}".')


def install_dependencies(args, project_root_dir):
    def eigen():
        build_dir = project_root_dir / "build" / "eigen"
        extracted_dir = find_dependency("eigen", project_root_dir)

        cmds = f"""
            mkdir -p "{build_dir}";
            cd "{build_dir}";
            cmake -DCMAKE_INSTALL_PREFIX="{project_root_dir}" \
                  -DCMAKE_INSTALL_DATADIR="{project_root_dir}/lib64" \
               {f'-DCMAKE_C_COMPILER="{args.C_compiler}"' if (args.C_compiler is not None) else ''} \
               {f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}"' if (args.CXX_compiler is not None) else ''} \
                  "{extracted_dir}";
            make install;
            """
        return cmds

    def pybind11():
        build_dir = project_root_dir / "build" / "pybind11"
        extracted_dir = find_dependency("pybind11", project_root_dir)

        cmds = f"""
            mkdir -p "{build_dir}";
            cd "{build_dir}";
            cmake -DCMAKE_INSTALL_PREFIX="{project_root_dir}" \
                  -DPYBIND11_INSTALL=ON \
                  -DPYBIND11_TEST=OFF \
                  -DPYBIND11_CMAKECONFIG_INSTALL_DIR="{project_root_dir}/lib64/cmake/pybind11" \
               {f'-DCMAKE_C_COMPILER="{args.C_compiler}"' if (args.C_compiler is not None) else ''} \
               {f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}"' if (args.CXX_compiler is not None) else ''} \
                  "{extracted_dir}";
            make install;
            """
        return cmds

    def fftw():
        build_dir = project_root_dir / "build" / "fftw"
        extracted_dir = find_dependency("fftw", project_root_dir)

        # FFTW's cmake interface cannot build float/double libraries at the same time, hence we
        # have to relaunch the commands with 2 different values for ENABLE_FLOAT.
        gen_cmd = (
            lambda compile_float: f"""
            mkdir -p "{build_dir}";
            cd "{build_dir}";
            cmake -DCMAKE_INSTALL_PREFIX="{project_root_dir}" \
                  -DENABLE_OPENMP={str(args.OpenMP).upper()} \
                  -DENABLE_THREADS={str(args.OpenMP).upper()} \
                  -DENABLE_FLOAT={compile_float} \
                  -DENABLE_SSE=ON \
                  -DENABLE_SSE2=ON \
                  -DENABLE_AVX=ON \
                  -DENABLE_AVX2=ON \
                  -DDISABLE_FORTRAN=ON \
               {f'-DCMAKE_C_COMPILER="{args.C_compiler}"' if (args.C_compiler is not None) else ''} \
               {f'-DCMAKE_CXX_COMPILER="{args.CXX_compiler}"' if (args.CXX_compiler is not None) else ''} \
                  "{extracted_dir}";
            make install;
            """
        )

        cmds = "\n".join([gen_cmd("OFF"), gen_cmd("ON")])
        return cmds

    cmds = "\n".join(
        [f'source "{project_root_dir}/pypeline.sh" --no_shell;', eigen(), pybind11(), fftw()]
    )
    return cmds


if __name__ == "__main__":
    project_root_dir = pathlib.Path(__file__).parent.absolute()

    args = parse_args()
    if args.lib is not None:
        cmds = build_lib(args, project_root_dir)
    elif args.doc is True:
        cmds = build_doc(args, project_root_dir)
    elif args.download_dependencies is True:
        cmds = download_dependencies(args, project_root_dir)
    elif args.install_dependencies is True:
        cmds = install_dependencies(args, project_root_dir)
    else:
        raise ValueError("Something went wrong.")

    if args.print is True:
        print(cmds)
    else:
        status = subprocess.run(
            cmds, stdin=None, stdout=sys.stdout, stderr=sys.stderr, shell=True, cwd=project_root_dir
        )

        print("\nSummary\n=======")
        print("Success" if (status.returncode == 0) else "Failure")
        print(cmds)
