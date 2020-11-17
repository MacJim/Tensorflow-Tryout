set custom_cudnn_lib_dir (cat custom_cudnn_lib_dir.txt)
echo "Custom cuDNN library directory: `$custom_cudnn_lib_dir`"

switch $LD_LIBRARY_PATH
case "*:"
    set -gx LD_LIBRARY_PATH (string trim --right --chars=: $LD_LIBRARY_PATH)
    echo "Trimmed trailing commas of `LD_LIBRARY_PATH`"
end
# echo $LD_LIBRARY_PATH

switch $LD_LIBRARY_PATH
case "$custom_cudnn_lib_dir*"
    echo "Do nothing: `$custom_cudnn_lib_dir` is already in `LD_LIBRARY_PATH`!"
case '*'
    set old_ld_library_path $LD_LIBRARY_PATH
    set -gx LD_LIBRARY_PATH "$custom_cudnn_lib_dir:$LD_LIBRARY_PATH"
    echo "Updated `LD_LIBRARY_PATH` from `$old_ld_library_path` to `$LD_LIBRARY_PATH`."
end