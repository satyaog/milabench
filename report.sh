#!/bin/bash


function to_csv {
    _header=1

    while read d
    do
        grep '"return_code": 0' "$d"*.data >/dev/null || grep 'Terminating process because it ran for longer than' "$d"*.data >/dev/null
        if [[ "$?" != "0" ]]
        then
            1>&2 echo SKIPPING "$d"
            continue
        fi

        _dir=$(basename "$d")
        gpu=$(echo "$_dir" | cut -d"_" -f1)

        _i=3
        while [[ ! "$(echo "$_dir" | cut -d"_" -f"$_i" | cut -d"." -f1)" -gt "0" ]]
        do
            1>&2 echo "$_dir" | cut -d"_" -f"$_i" | cut -d"." -f1

            if [[ -z "$(echo "$_dir" | cut -d"_" -f"$_i" | cut -d"." -f1)" ]]
            then
                break
            fi
            _i=$(($_i + 1))
        done

        bench=$(echo "$_dir" | cut -d"_" -f2-$(($_i - 1)))
        batch_size=$(echo "$_dir" | cut -d"_" -f$_i | cut -d"." -f1)
        1>&2 echo "$d"
        1>&2 echo "$gpu"
        1>&2 echo "$bench"
        1>&2 echo "$batch_size"

        if [[ -z "$batch_size" ]]
        then
            1>&2 echo SKIPPING "$d"
            continue
        fi

        while read l
        do
            1>&2 echo "$l"

            bench=$(echo $l | cut -d"|" -f1)
            if [[ "$bench" == "bench"* ]]
            then
                if [[ -z "${_header}" ]]
                then
                    continue
                fi 
                l=$(echo "gpu" "|" $bench "|" "batch_size" "|" $(echo $l | cut -d"|" -f2-))
                unset _header
            else
                l=$(echo $gpu "|" $bench "|" $(printf "%09d" $batch_size) "|" $(echo $l | cut -d"|" -f2-))
            fi
            l=${l// | /,}
            1>&2 echo "$l"

            echo "$l"
        done < <(hatch run milabench report --runs "$d" | grep " | ")
    done
}

to_csv 2>log.err >log.csv < <(ls -d ~/DATA/milabench/runs/*/ | grep -Ev "failed|staging")

to_csv 2>log.failed.err >log.failed.csv < <(ls -d ~/DATA/milabench/runs/*/ | grep -E "failed|staging")
