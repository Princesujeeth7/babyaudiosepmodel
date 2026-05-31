# GitHub Upload Notes

Recommended upload:

```text
Upload/push the folder office_submission_minimal/
Do not upload office_submission_minimal.zip to GitHub unless using Git LFS.
```

Reason:

```text
GitHub normal file limit is 100 MB per file.
The zip file is larger than 100 MB.
```

The largest individual files inside the folder are:

```text
teacher_model/model.safetensors                       about 90 MB
student_model/final_student_phase8_mask_distill.pt    about 66 MB
student_model/final_student_phase8_weights_only.pt    about 22 MB
```

These individual files are below 100 MB, but for safer version control use Git LFS:

```bash
git lfs install
git lfs track "*.pt"
git lfs track "*.safetensors"
git add .gitattributes
```

Then add the project:

```bash
git add office_submission_minimal
git commit -m "Add lightweight BS-RoFormer RoPE replacement student"
git push
```

If Git LFS is not available, upload the folder through the GitHub web UI or keep model checkpoints in a release/drive link and commit only code/docs.
