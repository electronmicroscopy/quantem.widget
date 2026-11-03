"""The live region workflow uses the same bounds and scientific arrays as Python."""
import json
import numpy as np
import pytest
from quantem.widget import ShowPtycho


def test_fit_then_reconstruct_keeps_regions_and_phase_arrays():
    regions=[{'id':'a','label':'A','bounds':[8,40,12,44]},
             {'id':'b','label':'B','bounds':[40,72,48,80]}]
    phase=np.arange(32*32,dtype=np.float32).reshape(32,32)/1024
    received=[]
    def fit(selected):
        received.append(selected)
        return {'regions':selected,'fits':[{**r,'sample_tilt_mrad':[2.,-3.]} for r in selected],
                'images':{'A':phase,'B':phase+1},'sampling_A':.2}
    widget=ShowPtycho(np.zeros((96,96),np.float32),local_tilt={'scan_shape':[96,96],'regions':regions},
                      fit_regions=fit,reconstruct_regions=fit)
    for index,action in enumerate(('fit','reconstruct')):
        widget.local_tilt_request_json=json.dumps({'id':index,'action':action,'signature':'selection','regions':regions})
        assert json.loads(widget.local_tilt_reply_json)=={'id':index,'signature':'selection','status':'complete','action':action}
        assert received[-1]==regions
        np.testing.assert_array_equal(np.frombuffer(widget.local_tilt_bytes,np.float32).reshape(2,32,32),[phase,phase+1])
    metadata=json.loads(widget.local_tilt_images_json)
    assert metadata[1]['offset']==phase.size
    assert metadata[0]['sampling_A']==.2


def test_changed_selection_carries_its_own_reply_and_saved_view_is_read_only():
    regions=[{'id':'a','label':'A','bounds':[0,32,0,32]}]
    widget=ShowPtycho(np.zeros((96,96)),local_tilt={'scan_shape':[96,96],'regions':regions})
    widget.set_region_images({'A':np.ones((32,32),np.float32)},sampling_A=.3)
    widget.local_tilt_request_json=json.dumps({'id':7,'signature':'edited','action':'fit','regions':regions})
    reply=json.loads(widget.local_tilt_reply_json)
    assert reply['status']=='error' and reply['signature']=='edited'
    assert 'Run the workspace cell' in reply['message']
    np.testing.assert_array_equal(np.frombuffer(widget.local_tilt_bytes,np.float32),1)


def test_delete_regions_keeps_other_fits_and_can_leave_an_empty_workspace():
    """Delete a fitted region, then the last rectangle, without recalculating."""
    regions = [{'id': 'a', 'label': 'A', 'bounds': [0, 32, 0, 32]},
               {'id': 'b', 'label': 'B', 'bounds': [32, 64, 32, 64]}]
    phase = np.arange(1024, dtype=np.float32).reshape(32, 32)
    widget = ShowPtycho(np.zeros((64, 64)), local_tilt={
        'scan_shape': [64, 64], 'regions': regions,
        'fits': [{**region, 'sample_tilt_mrad': [1, 2]} for region in regions]})
    widget.set_region_images({'A · SSB': phase, 'B · SSB': phase + 1}, sampling_A=.2)
    overview = widget.phase_bytes
    widget.local_tilt_active_region = 'a'
    widget.regions = regions[1:]
    assert widget.local_tilt_active_region == ''
    assert [fit['id'] for fit in widget.region_fits] == ['b']
    assert list(widget.region_images) == ['B · SSB']
    np.testing.assert_array_equal(widget.region_images['B · SSB'], phase + 1)
    assert widget.local_tilt_request_json == '' and widget.phase_bytes == overview
    widget.regions = []
    assert widget.region_fits == [] and widget.region_images == {}
    assert widget.local_tilt_images_json == '[]'
    widget.regions = regions[:1]
    assert widget.regions == regions[:1]


def test_pair_preserves_native_grids_and_per_image_physical_sampling():
    from quantem.widget.showptycho.local_tilt import _images
    ssb=np.arange(32*32,dtype=np.float32).reshape(32,32)
    ptycho=np.arange(64*64,dtype=np.float32).reshape(64,64)
    metadata,pixels=_images({'SSB':ssb,'Ptychography':ptycho},{'SSB':.26,'Ptychography':.13})
    first,second=json.loads(metadata)
    assert first['shape']==[32,32] and second['shape']==[64,64]
    assert first['shape'][0]*first['sampling_A']==second['shape'][0]*second['sampling_A']
    decoded=np.frombuffer(pixels,np.float32)
    np.testing.assert_array_equal(decoded[:ssb.size].reshape(ssb.shape),ssb)
    np.testing.assert_array_equal(decoded[second['offset']:].reshape(ptycho.shape),ptycho)


def test_invalid_sampling_is_rejected_before_transport():
    import pytest
    from quantem.widget.showptycho.local_tilt import _images
    for sampling in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='positive image sampling'):
            _images({'SSB':np.ones((32,32))},sampling)


def test_notebook_cells_read_edit_fit_and_reconstruct_current_regions():
    import pytest

    regions = [{'id': 'a', 'label': 'A', 'bounds': [0, 32, 0, 32]}]
    phase = np.full((32, 32), 1 + 2**-40, dtype=np.float64)
    def fit(selected):
        return {'regions': selected, 'fits': [{**r, 'sample_tilt_mrad': [2., -3.]} for r in selected],
                'images': {'A': phase}, 'sampling_A': .2}
    widget = ShowPtycho(np.zeros((96, 96)),
                       local_tilt={'scan_shape': [96, 96], 'regions': regions},
                       fit_regions=fit, reconstruct_regions=fit)
    assert widget.regions == regions
    with pytest.raises(ValueError, match='unfitted'):
        widget.region_fits
    fits = widget.fit_regions()
    assert fits[0]['sample_tilt_mrad'] == [2., -3.]
    np.testing.assert_array_equal(widget.region_images['A'], phase)
    assert widget.region_images['A'].dtype == np.float64
    # Returned values are snapshots, not mutable aliases to viewer state.
    fits[0]['bounds'][0] = 9
    copied = widget.region_images
    copied['A'][:] = 0
    assert widget.region_fits[0]['bounds'][0] == 0
    np.testing.assert_array_equal(widget.region_images['A'], phase)
    # A UI edit reaches Python even before fitting.
    changed = [{'id': 'a', 'label': 'A', 'bounds': [16, 48, 8, 40]}]
    widget.local_tilt_selection_json = json.dumps(changed)
    assert widget.regions == changed
    with pytest.raises(ValueError, match='unfitted'):
        widget.region_fits
    with pytest.raises(ValueError, match='another selection'):
        widget.region_images
    widget.fit_regions()
    assert widget.region_fits[0]['bounds'] == changed[0]['bounds']
    np.testing.assert_array_equal(widget.reconstruct_regions()['A'], phase)
    # Programmatic changes go through the same synchronized selection trait.
    widget.regions = regions
    assert json.loads(widget.local_tilt_selection_json) == regions
    with pytest.raises(ValueError, match='unfitted'):
        widget.region_fits


def test_python_callback_errors_never_return_previous_success():
    import pytest
    widget = ShowPtycho(np.zeros((64, 64)), local_tilt={
        'scan_shape': [64, 64], 'regions': [{'id': 'a', 'label': 'A', 'bounds': [0, 32, 0, 32]}]})
    with pytest.raises(RuntimeError, match='no live callback'):
        widget.fit_regions()
    ordinary = ShowPtycho(np.ones((32, 32)))
    assert ordinary.regions == []
    with pytest.raises(ValueError, match='Configure local_tilt'):
        ordinary.regions = widget.regions


def test_invalid_callback_images_preserve_the_previous_valid_fit():
    import pytest

    regions = [{'id': 'a', 'label': 'A', 'bounds': [0, 32, 0, 32]}]
    fits = [{**regions[0], 'sample_tilt_mrad': [1., 2.]}]
    def invalid_fit(selected):
        return {'regions': selected, 'fits': [{**selected[0], 'sample_tilt_mrad': [9., 9.]}],
                'images': {'A': np.full((32, 32), np.nan)}, 'sampling_A': .2}
    widget = ShowPtycho(np.zeros((64, 64)), local_tilt={
        'scan_shape': [64, 64], 'regions': regions, 'fits': fits}, fit_regions=invalid_fit)
    widget.set_region_images({'A': np.ones((32, 32))}, sampling_A=.2)
    with pytest.raises(RuntimeError, match='finite two-dimensional'):
        widget.fit_regions()
    assert widget.region_fits == fits
    np.testing.assert_array_equal(widget.region_images['A'], 1)


class _Fit:
    """One seed's joint fit; the tilt identifies which seed was kept."""
    def __init__(self, agreement, seed):
        self.aberrations = {'C10_nm': 3.0, 'C12_nm': 1.0, 'phi12_deg': 10.0}
        self.sample_tilt_mrad = (float(seed), 0.0)
        self.depth_spread_nm = 10.0
        self.sample_agreement = agreement
        self.trial_records = ()
        self.refinement = 'newton'


class _PaddedSession:
    """quantem.gpu.SSB as fit_regions reads it: a 96 x 80 scan centred in a 128 x 128 grid."""
    scan_region, source_path, backend = None, 'scan.npy', 'mps'
    voltage_kV, semiangle_mrad, scan_sampling_A = 300.0, 20.0, (0.5, 0.5)
    detector_sampling_mrad, bright_field_center_px, bright_field_radius_px = (5.0, 5.0), (7.5, 7.5), 4.0
    bright_field_intensity_threshold, scan_detector_rotation_flipped, num_bright_field_pixels = 0.5, False, 50
    scan_shape, input_scan_shape = (128, 128), (96, 80)

    def __init__(self, agreements):
        self.agreements = list(agreements)
        self.seeds = []

    def select_region(self, bounds, **options):
        import contextlib
        return contextlib.nullcontext(self)

    def find_aberrations(self, *, seed, **options):
        self.seeds.append(seed)
        self.options = options
        return _Fit(self.agreements[len(self.seeds) - 1], seed)

    def preview(self, aberrations, *, upsampling_factor, **options):
        rows, cols = np.indices((128 * upsampling_factor,) * 2)
        return (rows * 10000 + cols).astype(np.float32), None


def _fit_padded(agreements):
    from quantem.widget.adapters.gpu import SsbSession
    region = {'id': 'r', 'label': 'R', 'bounds': [8, 40, 4, 36]}
    parameters = {'aberrations': {'C10_nm': 3.0, 'C12_nm': 1.0, 'phi12_deg': 10.0},
                  'scan_detector_rotation_deg': 0.0}
    return SsbSession(_PaddedSession(agreements)).fit_regions(
        [region], parameters, upsampling_factor=2, phase_estimator='phase_of_mean')


def test_regional_fit_keeps_the_best_finite_seed_and_crops_measured_positions():
    pytest.importorskip("quantem.gpu", reason="the regional fit runs quantem.gpu's SSB")
    result = _fit_padded([float('nan'), 0.2, 0.9, float('inf'), 0.5])
    fit, = result['fits']
    assert fit['seed'] == 47 and fit['sample_tilt_mrad'] == [47.0, 0.0]  # third seed, agreement 0.9
    # Native bounds shift by the (16, 24) padding before scaling by the factor.
    rows, cols = np.indices((256, 256))
    full = (rows * 10000 + cols).astype(np.float32)
    np.testing.assert_array_equal(result['images']['R · SSB'], full[(8 + 16) * 2:(40 + 16) * 2, (4 + 24) * 2:(36 + 24) * 2])
    assert fit['reconstruction']['scan_offset'] == [16, 24]
    with pytest.raises(ValueError, match='nonfinite fit'):
        _fit_padded([float('nan')] * 5)


def test_regional_fit_passes_its_trial_count_refinement_and_tilt_limit_to_the_fit():
    pytest.importorskip("quantem.gpu", reason="the regional fit runs quantem.gpu's SSB")
    # Fit Regions hard-coded 200 trials and Newton and tested a copied 24.99 mrad tilt bound
    from quantem.widget.adapters.gpu import REGION_TILT_LIMIT_MRAD, SsbSession
    session = _PaddedSession([0.5] * 5)
    region = {'id': 'r', 'label': 'R', 'bounds': [8, 40, 4, 36]}
    parameters = {'aberrations': {'C10_nm': 3.0, 'C12_nm': 1.0, 'phi12_deg': 10.0}, 'scan_detector_rotation_deg': 0.0}
    result = SsbSession(session).fit_regions([region], parameters, upsampling_factor=1, phase_estimator='phase_of_mean',
                                             num_trials=12, refinement='nelder-mead')
    assert (session.options['num_trials'], session.options['refinement']) == (12, 'nelder-mead')
    assert session.options['tilt_limit_mrad'] == REGION_TILT_LIMIT_MRAD
    fit, = result['fits']
    assert (fit['fit_record']['num_trials'], fit['fit_record']['refinement']) == (12, 'nelder-mead')
    assert not fit['bound_hit']  # tilt (11, 0) mrad and depth 10 nm lie inside the limit and DEPTH_SPREAD_RANGE_NM


def test_regional_fit_flags_a_depth_at_the_end_of_quantem_gpus_search_range(monkeypatch):
    pytest.importorskip("quantem.gpu", reason="the regional fit runs quantem.gpu's SSB")
    from quantem.gpu.ssb import DEPTH_SPREAD_RANGE_NM
    original = _Fit.__init__

    def at_the_bound(self, agreement, seed):
        original(self, agreement, seed)
        self.depth_spread_nm = DEPTH_SPREAD_RANGE_NM[1]
    monkeypatch.setattr(_Fit, '__init__', at_the_bound)
    fit, = _fit_padded([0.5] * 5)['fits']
    assert fit['bound_hit'] and fit['depth_spread_nm'] == DEPTH_SPREAD_RANGE_NM[1]
