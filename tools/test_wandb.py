#!/usr/bin/env python3
"""Run test inference and upload predictions to W&B.

This script is a wrapper around ``tools/test.py``:
1) runs inference and writes predictions to a pickle file
2) uploads predictions (and eval json if present) to W&B artifact
"""

import argparse
import glob
import json
import os
import os.path as osp
import re
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run tools/test.py and log outputs to W&B')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        default=None,
        help='directory for test outputs (defaults under work_dirs)')
    parser.add_argument(
        '--out',
        default=None,
        help='prediction output pickle path (defaults to <work-dir>/predictions.pkl)'
    )
    parser.add_argument(
        '--wandb-project',
        default='mmdet_test_inference',
        help='W&B project name')
    parser.add_argument('--wandb-entity', default=None, help='W&B entity/team')
    parser.add_argument('--wandb-run-name', default=None, help='W&B run name')
    parser.add_argument(
        '--wandb-artifact-name',
        default=None,
        help='artifact name (defaults from config basename)')
    parser.add_argument(
        '--wandb-job-type',
        default='test_inference',
        help='W&B job type')
    parser.add_argument(
        '--wandb-tags',
        nargs='*',
        default=None,
        help='optional W&B tags')
    parser.add_argument('--wandb-notes', default=None, help='optional W&B notes')
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def _repo_root():
    return osp.dirname(osp.dirname(osp.abspath(__file__)))


def _sanitize_name(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', name).strip('-') or 'artifact'


def _default_work_dir(config_path):
    config_name = osp.splitext(osp.basename(config_path))[0]
    ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    return osp.join(_repo_root(), 'work_dirs', config_name, f'test_wandb_{ts}')


def _parse_show_dir(passthrough):
    for i, arg in enumerate(passthrough):
        if arg == '--show-dir' and i + 1 < len(passthrough):
            return passthrough[i + 1]
        if arg.startswith('--show-dir='):
            return arg.split('=', 1)[1]
    return None


def _run_test(config, checkpoint, work_dir, out_file, passthrough):
    cmd = [
        sys.executable,
        osp.join(_repo_root(), 'tools', 'test.py'),
        config,
        checkpoint,
        '--work-dir',
        work_dir,
        '--out',
        out_file,
    ]
    cmd.extend(passthrough)

    print('Running test command:')
    print('  ' + ' '.join([subprocess.list2cmdline([x]) for x in cmd]))
    subprocess.run(cmd, check=True)
    return cmd


def _collect_eval_json(work_dir):
    return sorted(glob.glob(osp.join(work_dir, 'eval_*.json')))


def _log_to_wandb(args, cmd, out_file, work_dir, eval_json_paths, show_dir):
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            'wandb is not installed in the current environment. '
            'Install wandb to use this script.') from exc

    config_base = osp.splitext(osp.basename(args.config))[0]
    ckpt_base = osp.splitext(osp.basename(args.checkpoint))[0]
    run_name = args.wandb_run_name or f'{config_base}-{ckpt_base}-test'
    artifact_name = args.wandb_artifact_name or f'{config_base}-test-predictions'
    artifact_name = _sanitize_name(artifact_name)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        job_type=args.wandb_job_type,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        config={
            'config_path': osp.abspath(args.config),
            'checkpoint_path': osp.abspath(args.checkpoint),
            'work_dir': osp.abspath(work_dir),
            'out_file': osp.abspath(out_file),
            'command': cmd,
        })

    for eval_json in eval_json_paths:
        try:
            with open(eval_json, 'r', encoding='utf-8') as f:
                eval_payload = json.load(f)
            metric = eval_payload.get('metric')
            if isinstance(metric, dict):
                run.log(metric)
        except Exception as exc:  # pragma: no cover
            print(f'Warning: failed to parse {eval_json}: {exc}')

    artifact = wandb.Artifact(
        artifact_name,
        type='inference_predictions',
        metadata={
            'config': osp.abspath(args.config),
            'checkpoint': osp.abspath(args.checkpoint),
            'work_dir': osp.abspath(work_dir),
        })

    if osp.isfile(out_file):
        artifact.add_file(out_file, name='predictions.pkl')
    else:
        raise FileNotFoundError(f'Prediction output file not found: {out_file}')

    for eval_json in eval_json_paths:
        artifact.add_file(eval_json, name=osp.join('eval', osp.basename(eval_json)))

    if show_dir and osp.isdir(show_dir):
        artifact.add_dir(show_dir, name='show_dir')

    run.log_artifact(artifact)
    run.finish()

    print(f'Logged W&B artifact "{artifact_name}" to project "{args.wandb_project}".')


def main():
    args, passthrough = parse_args()

    work_dir = args.work_dir or _default_work_dir(args.config)
    out_file = args.out or osp.join(work_dir, 'predictions.pkl')

    os.makedirs(work_dir, exist_ok=True)
    out_parent = osp.dirname(osp.abspath(out_file))
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    cmd = _run_test(args.config, args.checkpoint, work_dir, out_file, passthrough)
    eval_json_paths = _collect_eval_json(work_dir)
    show_dir = _parse_show_dir(passthrough)
    _log_to_wandb(args, cmd, out_file, work_dir, eval_json_paths, show_dir)


if __name__ == '__main__':
    main()
