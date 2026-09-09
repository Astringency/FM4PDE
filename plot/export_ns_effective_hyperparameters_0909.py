"""Update NS parameter provenance from all 450 completed runtime batches."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json

import torch

from run_ns_main_revision_0909 import EVAL, SETTINGS, configuration, digest


def read(path):
    return json.loads(path.read_text())


def canonical_sha(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def setting_for(row):
    if row['observations'] == 'full':
        assert row['task'] in ('forward', 'inverse')
        return 'full_' + row['task']
    assert row['observations'] == 'random'
    return {'forward': 'sparse_forward', 'inverse': 'sparse_inverse', 'both': 'sparse_joint'}[row['task']]


def main(args):
    torch.set_num_threads(2)
    args.audit.mkdir(parents=True, exist_ok=True)
    source = args.paper/'source_data'
    parameter_csv = source/'main_hyperparameters_verified.csv'
    original = args.audit/'main_hyperparameters_before.csv'
    if not original.exists():
        original.write_bytes(parameter_csv.read_bytes())
    previous = list(csv.DictReader(original.open(newline='')))
    old_fields = list(previous[0])
    assert len(previous) == 66 and sum(r['pde'] == 'nsnonbounded' for r in previous) == 15
    protected = {str(source/name): digest(source/name) for name in
                 ['main_configs_archive.json.gz', 'main_gate_trace_audit.json.gz']}
    frozen_handoff = read(args.study/'FINAL_HANDOFF.json')
    assert frozen_handoff['status'] == 'complete_and_validated'
    for name in ['collection_complete.json', 'tables/ns_main_summary.csv', 'tables/ns_main_per_sample.csv',
                 'tables/ns_main_validation.json', 'tables/ns_main_residual_audit.json']:
        path = args.study/name
        assert digest(path) == frozen_handoff['artifact_sha256'][str(path)]
    validation = read(args.study/'tables/ns_main_validation.json')
    residual = read(args.study/'tables/ns_main_residual_audit.json')
    assert validation['status'] == residual['status'] == 'pass'
    assert residual['complete'] and residual['examples'] == 15000 and residual['batches'] == 450
    assert all(residual[k] for k in ['all_frozen_configurations_match', 'all_truths_equal_frozen_source_cache',
        'all_protocol_and_selection_hashes_match', 'all_observation_counts_and_masks_match',
        'all_runtime_batches_and_device_assignments_match'])
    protocol_path, selection_path = args.study/'inputs/protocol.json', args.study/'selection.json'
    protocol, selection = read(protocol_path), read(selection_path)
    ph, sh = digest(protocol_path), digest(selection_path)
    weight_path = (args.study/'inputs/weights.pth').resolve()
    wh = digest(weight_path)
    assert ph == residual['protocol_sha256'] == selection['protocol_sha256']
    assert sh == residual['selection_sha256'] and wh == residual['model_weights_sha256'] == protocol['model']['weights_sha256']
    prepared_path = args.paper/'audit/revision_0909/ns_effective_settings/effective_settings.json'
    prepared = {(r['distribution'], r['setting']): r for r in read(prepared_path)['records']}
    environments = {i: read(args.study/f'main_results/environment_run_{i}.json') for i in (0, 1)}
    origins = {
        0: dict(host='server197', root='/research_data/users/zhangxifeng/C01Python/NSMainRevision0909/main_results'),
        1: dict(host='server193', root='/home/zhangxf/C01Python/NSMainWorker0909/main_results'),
    }
    producer = 'a409ba5522b8b71dbca940d640d4554119aef7c5'
    producer_hashes = {env['script_sha256'] for env in environments.values()}
    assert len(producer_hashes) == 1
    for env in environments.values():
        assert env['commit'] == producer and env['protocol_sha256'] == ph and env['tf32'] is True
        assert env['precision'] == 'float32'
    batches, cells, configurations = [], defaultdict(list), {}
    for path in sorted((args.study/'main_results').rglob('offset*.pt')):
        receipt_path = path.with_suffix('.json')
        receipt = read(receipt_path)
        assert digest(path) == receipt['result_sha256']
        assert receipt['protocol_sha256'] == ph and receipt['selection_sha256'] == sh
        data = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        ids, worker = receipt['ids'], receipt['worker']
        dist, setting = receipt['dist'], receipt['setting']
        key = dist, setting
        assert len(ids) == len(receipt['rows']) == receipt['batch_size'] and receipt['nfe'] == 100
        assert ids == list(range(ids[0], ids[-1] + 1)) and receipt['tf32'] is True
        for name in ['predictions', 'truths', 'masks']:
            assert tuple(data[name].shape) == (len(ids), 2, 128, 128)
        if worker == 0:
            assert all(2000 <= i < 2700 for i in ids) and len(ids) in (64, 60)
        else:
            assert worker == 1 and all(2700 <= i < 3000 for i in ids) and len(ids) in (16, 12)
        env = environments[worker]
        assert receipt['gpu'] == env['gpu']
        cfg = data['config']
        expected = configuration(protocol, setting, dist, cfg['checkpoint_path'], selection['scaled_inverse'].get(setting, False))
        expected.offset = ids[0]
        assert cfg == expected.asdict()
        assert all(cfg[k] == v for k, v in prepared[key]['effective_configuration'].items())
        normalized = {k: v for k, v in cfg.items() if k != 'offset'}
        if key in configurations:
            assert configurations[key] == normalized
        else:
            configurations[key] = normalized
        relative = str(path.relative_to(args.study/'main_results'))
        row = dict(distribution=dist, setting=setting, task=cfg['task'], first_index=ids[0], last_index=ids[-1],
            sample_indices=ids, runtime_batch_size=len(ids), saved_configuration_batch_size=cfg['batch_size'],
            worker=worker, environment_id=f'worker_{worker}', host_alias=origins[worker]['host'],
            gpu_model=env['gpu'], gpu_uuid=env['gpu_uuid'], torch=env['torch'], cuda=env['cuda'],
            precision=env['precision'], tf32_enabled=env['tf32'], nfe_per_prediction=receipt['nfe'],
            original_remote_result=origins[worker]['root']+'/'+relative,
            local_result=str(path), result_relative_to_main_results=relative, result_sha256=receipt['result_sha256'],
            receipt_sha256=digest(receipt_path), configuration_sha256=canonical_sha(cfg),
            configuration_without_offset_sha256=canonical_sha(normalized), protocol_sha256=ph,
            selection_sha256=sh, checkpoint_sha256=wh, producer_commit=producer, producer_script_sha256=env['script_sha256'])
        batches.append(row)
        cells[key].append(row)
    assert len(batches) == 450 and len(cells) == 15
    records = []
    for (dist, setting), group in sorted(cells.items()):
        ids = [i for r in group for i in r['sample_indices']]
        sizes = Counter(r['runtime_batch_size'] for r in group)
        worker_samples = Counter()
        worker_batches = Counter()
        for r in group:
            worker_samples[r['worker']] += r['runtime_batch_size']
            worker_batches[r['worker']] += 1
        assert sorted(ids) == EVAL and len(group) == 30
        assert sizes == {12: 1, 16: 18, 60: 1, 64: 10}
        assert worker_samples == {0: 700, 1: 300} and worker_batches == {0: 11, 1: 19}
        records.append(dict(distribution=dist, setting=setting, examples=1000, batches=30,
            runtime_batch_size_counts=dict(sorted(sizes.items())), runtime_samples_per_worker=dict(worker_samples),
            runtime_batches_per_worker=dict(worker_batches), configuration_without_offset=configurations[dist, setting],
            configuration_without_offset_sha256=canonical_sha(configurations[dist, setting])))
    model = copy.deepcopy(protocol['model'])
    model['resolved_inference_checkpoint'] = str(weight_path)
    model['normalizer_sha256'] = canonical_sha(model['normalizer'])
    model['effective_value_fourier_features'] = model['model_config']['with_value_fourier_features']
    model['effective_coordinate_fourier_features'] = model['model_config']['with_coordinate_fourier_features']
    config_file = source/'ns_main_effective_configurations_0909.json'
    batch_file = source/'ns_main_runtime_batches_0909.json'
    effective = dict(status='pass', examples=15000, cells=15, raw_batches=450, model=model,
        protocol_sha256=ph, selection_sha256=sh, producer_commit=producer,
        environments={f'worker_{i}': env for i, env in environments.items()}, records=records,
        semantics={'saved_configuration_batch_size': 'Native scalar configuration field; not the actual runtime batch size.',
                   'configuration_without_offset': 'All fields are identical across the 30 batches of a cell except the physical test offset.',
                   'model_flags': 'Boolean model configuration flags determine Fourier features; descriptive historical notes are retained as source metadata.'})
    batch_manifest = dict(status='pass', study_root=str(args.study), raw_results_root=str(args.study/'main_results'),
        producer_commit=producer, protocol_sha256=ph, selection_sha256=sh, checkpoint_sha256=wh,
        configuration_file=config_file.name, batches=batches)
    new_fields = ['revision_setting', 'record_source_kind', 'runtime_batch_source', 'runtime_batch_size_counts',
        'runtime_gpu_models', 'runtime_gpu_uuids', 'runtime_samples_per_worker', 'runtime_batches_per_worker',
        'runtime_environment_ids', 'tf32_enabled', 'saved_configuration_batch_size', 'model_profile',
        'model_parameter_count', 'checkpoint_sha256', 'recorded_checkpoint_path', 'model_inference_signature',
        'normalizer_sha256', 'protocol_sha256', 'selection_sha256', 'sampling_source_commit',
        'sampling_source_sha256', 'effective_configurations_file', 'batch_manifest_file', 'test_index_range',
        'gradient_target', 'obs_guidance_reduction', 'pde_guidance_reduction', 'ns_operator_mode', 'noise_level']
    updated = copy.deepcopy(previous)
    record_index = {(r['distribution'], r['setting']): r for r in records}
    changed = []
    for row in updated:
        row.update({field: '' for field in new_fields})
        if row['pde'] != 'nsnonbounded':
            continue
        key = row['distribution'].lower(), setting_for(row)
        record, cfg = record_index[key], configurations[key]
        before = {field: row[field] for field in old_fields}
        for field in old_fields:
            if field in cfg:
                row[field] = str(cfg[field])
        row.update(source_root=str(args.study/'main_results'), sheet='', workbook_row='', n='1000',
            batch_sizes='12;16;60;64', archived_batches='30', checkpoint_path=str(weight_path),
            revision_setting=key[1], record_source_kind='validated_revision_0909_raw_batches',
            runtime_batch_source='receipt.batch_size', runtime_batch_size_counts=json.dumps(record['runtime_batch_size_counts'], sort_keys=True),
            runtime_gpu_models=';'.join(environments[i]['gpu'] for i in (0, 1)),
            runtime_gpu_uuids=';'.join(environments[i]['gpu_uuid'] for i in (0, 1)),
            runtime_samples_per_worker=json.dumps(record['runtime_samples_per_worker'], sort_keys=True),
            runtime_batches_per_worker=json.dumps(record['runtime_batches_per_worker'], sort_keys=True),
            runtime_environment_ids='worker_0;worker_1', tf32_enabled='True', saved_configuration_batch_size=str(cfg['batch_size']),
            model_profile=model['model_profile'], model_parameter_count=str(model['parameter_count']),
            checkpoint_sha256=wh, recorded_checkpoint_path=cfg['checkpoint_path'],
            model_inference_signature=model['inference_signature'], normalizer_sha256=model['normalizer_sha256'],
            protocol_sha256=ph, selection_sha256=sh, sampling_source_commit=producer,
            sampling_source_sha256=next(iter(producer_hashes)), effective_configurations_file=config_file.name,
            batch_manifest_file=batch_file.name, test_index_range='2000..2999',
            gradient_target=cfg['gradient_target'], obs_guidance_reduction=cfg['obs_guidance_reduction'],
            pde_guidance_reduction=cfg['pde_guidance_reduction'], ns_operator_mode=cfg['ns_operator_mode'],
            noise_level=str(cfg['noise_level']))
        changed.append(dict(key=list(key), before=before, after=row))
    assert len(changed) == 15
    assert all(all(old[k] == new[k] for k in old_fields) for old, new in zip(previous, updated) if old['pde'] != 'nsnonbounded')
    prior_audit = args.audit/'integration_audit.json'
    allowed_existing = {digest(original)}
    if prior_audit.exists():
        allowed_existing.add(read(prior_audit)['outputs'][str(parameter_csv)])
    assert digest(parameter_csv) in allowed_existing, 'Parameter CSV changed concurrently; refusing to overwrite.'
    save(config_file, effective)
    save(batch_file, batch_manifest)
    flat_batches = [{**r, 'sample_indices': ';'.join(map(str, r['sample_indices']))} for r in batches]
    write_csv(source/'ns_main_runtime_batches_0909.csv', flat_batches)
    write_csv(parameter_csv, updated, old_fields + new_fields)
    reread = list(csv.DictReader(parameter_csv.open(newline='')))
    assert len(reread) == 66 and all(all(old[k] == new[k] for k in old_fields)
        for old, new in zip(previous, reread) if old['pde'] != 'nsnonbounded')
    assert all(digest(Path(path)) == value for path, value in protected.items())
    outputs = [parameter_csv, config_file, batch_file, source/'ns_main_runtime_batches_0909.csv']
    save(prior_audit, dict(status='pass', ns_rows_updated=15, non_ns_rows_preserved=51,
        old_field_values_preserved_for_all_non_ns_rows=True, original_csv_sha256=digest(original),
        study_final_handoff_sha256=digest(args.study/'FINAL_HANDOFF.json'), prepared_settings_sha256=digest(prepared_path),
        cells=records, examples=15000, batches=450, all_raw_and_receipt_hash_bindings_verified=True,
        all_frozen_configurations_verified=True, all_runtime_partitions_verified=True,
        protected_archive_sha256=protected, changes=changed, outputs={str(p): digest(p) for p in outputs}))
    print(json.dumps(dict(status='pass', updated_ns_rows=15, preserved_other_rows=51, verified_batches=450,
                          verified_examples=15000, audit=str(prior_audit))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    main(parser.parse_args())
