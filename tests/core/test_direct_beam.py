"""Finite direct-beam oracle: ray/plane intersection, independent of fitter."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from xrd_tools.core.geometry.diffractometer import ImageOrientation
from xrd_tools.integrate.direct_beam import calibrate_direct_beam, calibrate_direct_beam_scan


def beam_images(angles, *, shape=(45, 161), pitches=(120e-6, 80e-6),
                distance=.08, zero=7., motion='horizontal', centre=None):
    # Lab ray (0,0,1) meets plane normal n at r = D*ray/(ray.n).
    # Its projection along the detector's rotated in-plane axis is the pixel shift.
    axis = 1 if motion == 'horizontal' else 0
    origin = (np.array(shape)[::-1] - 1) / 2 if centre is None else np.array(centre)
    expected, images = [], []
    yy, xx = np.indices(shape)
    for angle in angles:
        theta = np.deg2rad(angle-zero)
        normal = np.array([np.sin(theta), 0., np.cos(theta)])
        tangent = np.array([np.cos(theta), 0., -np.sin(theta)])
        ray = np.array([0., 0., 1.])
        hit = distance * ray / np.dot(ray, normal)
        pixel = origin[0 if axis == 1 else 1] + np.dot(hit, tangent) / pitches[axis]
        x, y = (pixel, origin[1]) if axis == 1 else (origin[0], pixel)
        image = 12 + .015*xx + .02*yy + 4000*np.exp(-.5*(((xx-x)/1.6)**2+((yy-y)/1.6)**2))
        images.append(image)
        expected.append(pixel)
    return images, np.array(expected), tuple(origin)


def config(shape=(45, 161), pitches=(120e-6, 80e-6)):
    return dict(pixel1=pitches[0], pixel2=pitches[1], max_shape=shape)


def run(angles, images, beam=(80, 22), **kwargs):
    return calibrate_direct_beam(angles, images, detector='Detector',
                               detector_config=config(), beam_pixel=beam,
                               angle_zero_deg=7., **kwargs)


@pytest.mark.parametrize('motion,shape,pitches', [
    ('horizontal', (45,161), (120e-6,80e-6)),
    ('vertical', (171,39), (55e-6,130e-6)),
])
def test_geometry_and_original_pairing(motion, shape, pitches):
    angles = np.array([8.1, 7.7, 7.13, 7., 6.82, 6.2, 5.7])
    images, expected, beam = beam_images(angles, shape=shape, pitches=pitches, motion=motion)
    originals = [im.copy() for im in images]
    images[2] = None
    images[4] = np.zeros(shape)
    for image in images:
        if image is not None:
            image.flags.writeable = False
    ids = [f'point-{k*3}' for k in range(len(images))]
    result = calibrate_direct_beam(angles, iter(images), detector='Detector',
        detector_config=config(shape,pitches), beam_pixel=beam, motion=motion,
        angle_zero_deg=7., frame_ids=ids)
    assert result.distance_m == pytest.approx(.08, rel=2e-5)
    assert result.distance_mm == pytest.approx(80., rel=2e-5)
    assert result.distance_std_m >= 0
    assert result.motion_sign == -1
    assert result.slope_deg_per_pixel < 0
    np.testing.assert_array_equal(result.angles_deg, angles)
    assert result.frame_ids == tuple(ids)
    np.testing.assert_array_equal(result.used, [1,1,0,1,0,1,1])
    np.testing.assert_allclose(result.centres_px[result.used], expected[result.used], atol=2e-4)
    np.testing.assert_allclose(result.residuals_px[result.used], 0, atol=2e-4)
    assert 'missing' in result.rejection_reasons[2]
    assert 'peak' in result.rejection_reasons[4]
    for index in [0,1,3,5,6]:
        np.testing.assert_array_equal(images[index], originals[index])


def test_pixel_zero_and_clipped_strip():
    angles = np.array([7., 6.8, 6.5, 6.1])
    images, expected, _ = beam_images(angles, centre=(0,0))
    result = run(angles, images, beam=(0,0))
    assert result.used.all()
    assert result.centres_px[0] == pytest.approx(0., abs=1e-4)
    assert result.distance_m == pytest.approx(.08, rel=2e-5)
    np.testing.assert_allclose(result.centres_px, expected, atol=2e-4)


@pytest.mark.parametrize('orientation', [ImageOrientation(rotation=90),
    ImageOrientation(transpose=True,flip_vertical=True),
    ImageOrientation(flip_horizontal=True), ImageOrientation(rotation=180)])
def test_orientation_binning_mask_roi(orientation):
    from pyFAI.detectors import Detector
    angles = np.array([8.,7.2,7.,6.3,6.])
    images, expected, beam = beam_images(angles)
    det = Detector(pixel1=60e-6, pixel2=40e-6, max_shape=(90,322))
    # Coordinates/mask/ROI supplied in the oriented frame. The strip seed is deliberately
    # approximate, so it must not fix the intercept geometrically.
    marker = np.zeros(images[0].shape); marker[22,80] = 1
    y,x = np.argwhere(orientation.apply(marker))[0]
    shape = orientation.apply(marker).shape
    mask = np.zeros(shape,bool); mask[0,0] = True
    result = calibrate_direct_beam(angles, images, detector=det,
        beam_pixel=(x,y), orientation=orientation,
        motion='vertical' if orientation.swaps_axes else 'horizontal',
        angle_zero_deg=7., mask=mask, roi=(0,0,shape[1],shape[0]))
    assert result.distance_m == pytest.approx(.08, rel=2e-5)
    assert result.pixel_pitch_m == pytest.approx(80e-6)
    assert result.image_shape == shape
    assert det.shape == (90,322) and det.pixel2 == 40e-6  # not mutated
    axis = 0 if orientation.swaps_axes else 1
    coordinate_image = np.broadcast_to(np.arange(161), (45,161))
    transformed = orientation.apply(coordinate_image)
    direction = np.sign(np.diff(transformed, axis=axis).mean())
    assert result.motion_sign == -direction


def test_oriented_mask_and_roi_exclude_the_correct_pixel():
    angles = [8.,7.5,7.,6.5,6.]
    images, _, _ = beam_images(angles)
    orientation = ImageOrientation(rotation=90)
    mask = np.zeros((161,45), bool)
    mask[80,22] = True
    result = run(angles,images,beam=(22,80),orientation=orientation,motion='vertical',
                 roi=(19,45,26,116),mask=mask)
    assert result.distance_m == pytest.approx(.08,rel=2e-5)
    np.testing.assert_array_equal(result.used,[1,1,0,1,1])


@pytest.mark.parametrize('damage', ['nan','negative','mask','saturation'])
def test_invalid_beam_core_rejected(damage):
    angles = np.array([8.,7.5,7.,6.5,6.])
    images, _, _ = beam_images(angles)
    kwargs = {}
    if damage == 'mask':
        mask = np.zeros(images[0].shape,bool); mask[:,80] = True
        kwargs['mask'] = mask
    elif damage == 'saturation':
        images[2][22,80] = 10000
        kwargs['saturation'] = 9000
    else:
        images[2][22,80] = np.nan if damage == 'nan' else -1
    result = run(angles, images, **kwargs)
    assert result.distance_m == pytest.approx(.08, rel=2e-5)
    assert not result.used[2]
    assert result.rejection_reasons[2]


def test_integer_saturation_and_bad_shape():
    angles = [8,7.5,7,6.5,6]
    images, _, _ = beam_images(angles)
    images[2] = images[2].astype('uint16'); images[2][22,80] = 65535
    images[3] = np.zeros((3,3))
    result = run(angles,images)
    assert 'saturat' in result.rejection_reasons[2]
    assert 'shape' in result.rejection_reasons[3]


@pytest.mark.parametrize('damage', ['static_only', 'saturation', 'core'])
def test_masked_uint32_sentinel_is_not_valid_saturation(damage):
    angles = [8., 7.5, 7., 6.5, 6.]
    images, _, _ = beam_images(angles)
    # Corrected counts above 65535 are valid; the acquisition cutoff is explicit.
    images = [(image * 25).astype('uint32') for image in images]
    mask = np.zeros(images[0].shape, bool)
    mask[22, 12] = True
    for image in images:
        image[22, 12] = np.iinfo('uint32').max
    expected = np.ones(5, bool)
    if damage == 'saturation':
        images[1][22, 20] = 200000  # Unmasked and away from the beam core.
        expected[1] = False
    elif damage == 'core':
        mask[22, 80] = True
        images[2][22, 80] = np.iinfo('uint32').max
        expected[2] = False
    result = run(angles, images, mask=mask, saturation=200000)
    np.testing.assert_array_equal(result.used, expected)
    assert result.distance_m == pytest.approx(.08, rel=1e-4)
    assert result.intercept_px == pytest.approx(80., abs=.001)
    if damage == 'saturation':
        assert 'saturated' in result.rejection_reasons[1]
    elif damage == 'core':
        assert 'masked/gap' in result.rejection_reasons[2]
    # Removing the static mask makes the very same sentinel invalid saturation.
    with pytest.raises(ValueError, match='saturated pixels'):
        run(angles, images, saturation=200000)


def test_linear_notebook_comparison():
    angles = np.array([7.15,7.08,7.,6.97,6.88])
    images, centres, _ = beam_images(angles)
    tangent = run(angles,images)
    linear = run(angles,images,model='linear')
    slope, intercept = np.polyfit(centres,angles,1)
    notebook = 80e-6/np.tan(abs(slope)*np.pi/180)
    assert linear.distance_m == pytest.approx(notebook,rel=2e-6)
    assert linear.distance_m == pytest.approx(tangent.distance_m,rel=1e-5)
    assert linear.slope_deg_per_pixel == pytest.approx(slope,rel=2e-6)


@pytest.mark.parametrize('angles,images,message', [
    ([7,7,7],[np.ones((45,161))]*3,'distinct'),
    ([6,7],[np.ones((45,161))]*2,'three'),
    ([6,np.nan,8],[np.ones((45,161))]*3,'finite'),
    ([6,7,8],[np.ones((45,161))]*3,'usable'),
    ([6,7,8],[], 'length'),
])
def test_invalid_scans(angles,images,message):
    with pytest.raises(ValueError,match=message): run(angles,images)


def test_geometry_requires_explicit_zero_and_uniform_flat_detector():
    from pyFAI.detectors import detector_factory
    with pytest.raises(ValueError,match='angle_zero'):
        calibrate_direct_beam([0,1,2],[],detector='Pilatus100k',beam_pixel=(243,97))
    for name in ['Aarhus','ImXPadS10']:
        with pytest.raises(ValueError,match='uniform.*flat|flat.*uniform'):
            calibrate_direct_beam([0,1,2],[],detector=detector_factory(name),
                                 beam_pixel=(0,0),angle_zero_deg=0)


def test_real_spec_raw_and_repetition(tmp_path):
    angles = [8.,7.2,7.,6.3,6.]
    images, _, _ = beam_images(angles)
    spec = tmp_path/'scan'
    spec.write_text('#F scan\n#O0 del\n\n#S 23 ascan del 0 1 4 1\n#P0 99\n#N 2\n#L del  I0\n'
                    +'\n'.join(f'{a} 1' for a in [0,0,0,0,0])
                    +'\n\n#S 23 ascan del 8 6 4 1\n#P0 99\n#N 2\n#L del  I0\n'
                    +'\n'.join(f'{a} 1' for a in angles)+'\n')
    for i,im in enumerate(images):
        if i != 2:
            with (tmp_path/f'beam_{i+1:03d}.raw').open('wb') as f:
                f.write(b'header!!'); f.write(im.astype('<i4').tobytes())
    result = calibrate_direct_beam_scan(spec,'23.2',image_dir=tmp_path,
        filename_template='beam_{point:03d}.raw',point_index_origin=1,
        detector='Detector',detector_config=config(),beam_pixel=(80,22),angle_zero_deg=7.,
        reader_options={'detector_shape':(45,161),'raw_dtype':'<i4','raw_header_skip':8})
    np.testing.assert_array_equal(result.angles_deg,angles)
    assert result.point_ids == tuple(('23.2',i) for i in range(1,6))
    assert result.frame_ids[2] == tmp_path/'beam_003.raw'
    assert 'FileNotFoundError' in result.rejection_reasons[2]
    assert result.distance_m == pytest.approx(.08,rel=1e-4)
    with pytest.raises(ValueError,match='scanned column'):
        calibrate_direct_beam_scan(spec,'23.2',images=images,motor='nu',
            detector='Detector',detector_config=config(),beam_pixel=(80,22),angle_zero_deg=7.)


@pytest.mark.parametrize('kind',['tiff','hdf5'])
def test_format_reader_and_exact_frame_references(tmp_path,kind):
    angles = [8.,7.2,7.,6.3,6.]
    images, _, _ = beam_images(angles)
    if kind == 'tiff':
        import tifffile
        refs=[]
        for i,image in enumerate(images):
            path=tmp_path/f'{i}.tiff'; tifffile.imwrite(path,image.astype('float32')); refs.append(path)
    else:
        import h5py
        path=tmp_path/'frames.h5'
        with h5py.File(path,'w') as f: f['custom/data']=np.array(images)
        refs=[(path,i,'/custom/data') for i in range(5)]
    result=run(angles,iter(refs))
    assert result.frame_ids == tuple(refs)
    assert result.distance_m == pytest.approx(.08,rel=2e-5)
    if kind == 'hdf5':
        refs[2]=(path,99,'/custom/data')
        result=run(angles,refs)
        assert not result.used[2] and result.rejection_reasons[2]


@pytest.mark.parametrize('layout', ['nexus', 'external_link', 'bluesky'])
def test_nexus_native_references_and_streams_preserve_selected_rows(tmp_path, layout):
    import h5py
    from xrd_tools.io.image import read_image
    from xrd_tools.io.nexus import open_nexus_image_stack
    from xrd_tools.sources import ContainerCursor

    angles = np.array([8., 7.5, 7., 6.5, 6.])
    images = np.asarray(beam_images(angles)[0])
    images = (images * 25).astype('uint32')
    mask = np.zeros(images.shape[1:], bool)
    mask[22, 12] = True
    images[:, 22, 12] = np.iinfo('uint32').max
    path = tmp_path / 'beam_master.h5'
    if layout == 'external_link':
        part = tmp_path / 'beam_data_000001.h5'
        with h5py.File(part, 'w') as f:
            f['entry/data/data'] = images
    with h5py.File(path, 'w') as f:
        entry = f.create_group('entry')
        entry.attrs['NX_class'] = 'NXentry'
        data = entry.create_group('data')
        data.attrs['NX_class'] = 'NXdata'
        data.attrs['signal'] = 'image'
        if layout == 'external_link':
            data['image'] = h5py.ExternalLink(part.name, '/entry/data/data')
            entry['instrument/detector/detectorSpecific/nimages'] = 1
        else:
            data['image'] = images
        if layout == 'bluesky':
            f.attrs['creator'] = 'NXWriter'
            entry['end_time'] = np.bytes_('2026-09-23T00:00:00')
            entry['instrument/bluesky/metadata/motors'] = np.bytes_('!!python/tuple\n- del\n')
            entry['instrument/positioners/del/value'] = angles
            data['del'] = angles
            data['image'].attrs['signal_type'] = 'detector'
    with open_nexus_image_stack(path) as stack:
        assert stack.shape == images.shape  # Actual frames, not header nimages.
    selected = (4, 0, 2)  # Deliberate reordered subset of a larger container.
    refs = tuple((path, i) for i in selected)
    points = tuple(('scan-8', i) for i in selected)
    kwargs = dict(mask=mask, saturation=200000, frame_ids=refs, point_ids=points)
    raw = run(angles[list(selected)], refs, **kwargs)
    masked = run(angles[list(selected)],
        (read_image(p, frame=i, mask=mask, exact_frame=True) for p, i in refs), **kwargs)
    with ContainerCursor(path) as cursor:
        motor = cursor.metadata_provider().motors()['del'] if layout == 'bluesky' else angles
        assert len(motor) == cursor.frame_count
        streamed = run(motor[list(selected)],
            (cursor.read_frame(i) for i in selected), **kwargs)
    for result in (raw, masked, streamed):
        assert result.used.all()
        assert result.frame_ids == refs and result.point_ids == points
        np.testing.assert_array_equal(result.angles_deg, angles[list(selected)])
        np.testing.assert_allclose(result.centres_px, raw.centres_px, atol=1e-6, rtol=0)
        assert result.distance_mm == pytest.approx(80., rel=1e-4)
    if layout == 'bluesky':
        automatic = calibrate_direct_beam_scan(path, detector='Detector', detector_config=config(),
            beam_pixel=(80, 22), angle_zero_deg=7., mask=mask, saturation=200000)
        assert automatic.used.all()
        np.testing.assert_array_equal(automatic.angles_deg, angles)


@pytest.fixture
def direct_beam_sources(tmp_path):
    import h5py
    angles = np.array([8., 7.5, 7., 6.5, 6.])
    images = (np.asarray(beam_images(angles)[0]) * 25).astype('uint32')
    images[:, 22, 12] = np.iinfo('uint32').max
    mask = np.zeros(images.shape[1:], bool)
    mask[22, 12] = True
    spec = tmp_path / 'beam.spec'
    spec.write_text('#F beam\n#O0 eta\n\n#S 8 ascan del 8 6 4 1\n#P0 0\n#N 1\n#L del\n'
                    + '\n'.join(str(a) for a in angles) + '\n')
    nexus = tmp_path / 'beam.nxs'
    with h5py.File(nexus, 'w') as f:
        f.create_group('entry').attrs['NX_class'] = 'NXentry'
        f['entry/data/del'] = angles
        f['entry/data/image'] = images
    master = tmp_path / 'beam_master.h5'
    with h5py.File(master, 'w') as f:
        f.create_group('entry').attrs['NX_class'] = 'NXentry'
        f['entry/data/data_000001'] = h5py.ExternalLink(nexus.name, '/entry/data/image')
        f['entry/instrument/detector/detectorSpecific/nimages'] = 1
    options = dict(detector='Detector', detector_config=config(),
                   beam_pixel=(71, 22), angle_zero_deg=7., mask=mask, saturation=200000)
    return spec, nexus, master, angles, options


@pytest.mark.parametrize('kind', ['spec_eiger', 'nexus', 'explicit_eiger'])
def test_single_scan_wrapper_loads_sources(direct_beam_sources, kind):
    spec, nexus, master, angles, options = direct_beam_sources
    if kind == 'spec_eiger':
        result = calibrate_direct_beam_scan(spec, '8.1', image_file=master, **options)
        assert result.point_ids == tuple(('8.1', i) for i in range(5))
    elif kind == 'nexus':
        result = calibrate_direct_beam_scan(nexus, **options)
    else:
        result = calibrate_direct_beam_scan(master, angles_deg=angles, **options)
    path = nexus if kind == 'nexus' else master
    assert result.frame_ids == tuple((path, i) for i in range(5))
    np.testing.assert_array_equal(result.angles_deg, angles)
    assert result.used.all()
    assert result.distance_mm == pytest.approx(80., rel=1e-4)
    assert result.intercept_px == pytest.approx(80., abs=.001)  # Free, not beam x=71.


def test_source_loader_is_lazy_closes_and_preserves_subset(direct_beam_sources):
    import h5py
    from xrd_tools.integrate import open_direct_beam_scan
    spec, nexus, master, angles, options = direct_beam_sources
    before = h5py.h5f.get_obj_count()
    with open_direct_beam_scan(spec, '8.1', image_file=master,
                               frame_indices=(4, 0, 2)) as inputs:
        frames = list(inputs.images)
        assert all(frame.image is None for frame in frames)
        np.testing.assert_array_equal(inputs.angles_deg, angles[[4, 0, 2]])
        assert inputs.frame_ids == tuple((master, i) for i in (4, 0, 2))
        assert inputs.point_ids == tuple(('8.1', i) for i in (4, 0, 2))
        result = calibrate_direct_beam(inputs.angles_deg, frames,
            frame_ids=inputs.frame_ids, point_ids=inputs.point_ids, **options)
        assert result.used.all()
    assert h5py.h5f.get_obj_count() == before
    with pytest.raises(ValueError):
        calibrate_direct_beam_scan(nexus, **{**options, 'beam_pixel': (-1, -1)})
    assert h5py.h5f.get_obj_count() == before


def test_source_wrapper_count_mismatch_and_explicit_subset(direct_beam_sources):
    import h5py
    spec, nexus, master, angles, options = direct_beam_sources
    with h5py.File(nexus, 'r+') as f:
        del f['entry/data/del']
        f['entry/data/del'] = angles[:4]
    with pytest.raises(ValueError, match='count'):
        calibrate_direct_beam_scan(nexus, **options)
    result = calibrate_direct_beam_scan(nexus, frame_indices=(3, 0, 2), **options)
    np.testing.assert_array_equal(result.angles_deg, angles[[3, 0, 2]])
    assert result.frame_ids == tuple((nexus, i) for i in (3, 0, 2))
    with pytest.raises(ValueError, match='angle'):
        calibrate_direct_beam_scan(nexus, frame_indices=(0, 2, 4), **options)


def test_source_wrapper_rejects_missing_motor_and_conflicting_inputs(direct_beam_sources):
    spec, nexus, master, angles, options = direct_beam_sources
    with pytest.raises(ValueError, match='del'):
        calibrate_direct_beam_scan(master, **options)
    with pytest.raises(ValueError, match='eta'):
        calibrate_direct_beam_scan(spec, 8, motor='eta', image_file=master, **options)
    with pytest.raises(ValueError, match='either|only one'):
        calibrate_direct_beam_scan(spec, 8, images=(), image_file=master, **options)


def test_frame_source_labels_metadata_and_read_rejections(tmp_path):
    import tifffile
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.sources import MemoryFrameSource
    angles = [8., 7.5, 7., 6.5, 6.]
    labels = (10, 20, 30, 40, 50)
    frames = []
    for label, angle, image in zip(labels, angles, beam_images(angles)[0]):
        path = tmp_path / f'{label}.tif'
        if label != 30:
            tifffile.imwrite(path, image.astype('float32'))
        frames.append(ScanFrame(index=label, source_path=path, metadata={'del': angle}))
    source = MemoryFrameSource(frames, name='run-uid')
    result = calibrate_direct_beam_scan(source, detector='Detector', detector_config=config(),
                                       beam_pixel=(80, 22), angle_zero_deg=7.)
    assert result.frame_ids == tuple(('run-uid', label) for label in labels)
    assert result.point_ids == result.frame_ids
    np.testing.assert_array_equal(result.angles_deg, angles)
    np.testing.assert_array_equal(result.used, [1, 1, 0, 1, 1])
    assert 'FileNotFoundError' in result.rejection_reasons[2]
    assert source.load_frame(10).shape == (45, 161)  # Caller still owns its source.


def test_caller_source_explicit_angles_and_original_labels():
    from xrd_tools.core.scan import ScanFrame
    from xrd_tools.sources import MemoryFrameSource
    angles = np.array([8., 7.5, 7., 6.5, 6.])
    labels = (10, 20, 30, 40, 50)
    source = MemoryFrameSource([ScanFrame(index=i, image=image)
        for i, image in zip(labels, beam_images(angles)[0])], name='remote-run-uid')
    result = calibrate_direct_beam_scan(source, angles_deg=angles, frame_indices=(50, 10, 30),
        detector='Detector', detector_config=config(), beam_pixel=(80, 22), angle_zero_deg=7.)
    np.testing.assert_array_equal(result.angles_deg, angles[[4, 0, 2]])
    assert result.frame_ids == tuple(('remote-run-uid', i) for i in (50, 10, 30))
    assert result.used.all()


def test_caller_cursor_remains_open_and_nondefault_entry(direct_beam_sources):
    import h5py
    from xrd_tools.sources import ContainerCursor, NexusStackSource
    _, nexus, master, angles, options = direct_beam_sources
    with ContainerCursor(master) as cursor:
        source = NexusStackSource(master, cursor=cursor)
        result = calibrate_direct_beam_scan(source, angles_deg=angles, **options)
        assert result.used.all()
        assert cursor.read_frame(0).dtype == np.dtype('uint32')
    with h5py.File(nexus, 'r+') as f:
        f.move('entry', 'measurement')
    result = calibrate_direct_beam_scan(nexus, entry='measurement', **options)
    assert result.used.all()
    np.testing.assert_array_equal(result.angles_deg, angles)


def test_nexus_angle_read_does_not_materialize_images(direct_beam_sources, monkeypatch):
    import h5py
    from xrd_tools.io.nexus import read_nexus
    _, nexus, _, angles, _ = direct_beam_sources
    reads = []
    original = h5py.Dataset.__array__
    def observed_read(dataset, *args, **kwargs):
        reads.append(dataset.name)
        return original(dataset, *args, **kwargs)  # Spy only; actual HDF5 reads.
    monkeypatch.setattr(h5py.Dataset, '__array__', observed_read)
    result = read_nexus(nexus, motor_names=['del'], counter_names=[])
    np.testing.assert_array_equal(result.angles['del'], angles)
    assert '/entry/data/image' not in reads


def test_import_is_lazy_and_headless():
    child = subprocess.run([sys.executable,'-c',
        'import sys; from xrd_tools.integrate import calibrate_direct_beam; '
        'assert not any(m.startswith(("pyFAI", "PyQt", "PySide", "xrayutilities")) for m in sys.modules)'],
        capture_output=True,text=True)
    assert child.returncode == 0,child.stderr


def test_registered_binning_and_unsigned_dummy():
    angles = np.array([1.,.5,0.,-.5,-1.])
    shape=(65,487)
    images,_,beam=beam_images(angles,shape=shape,pitches=(516e-6,172e-6),distance=.5,zero=0)
    images=[im.astype('uint32') for im in images]
    images[2][32,243]=np.iinfo('uint32').max-1  # Pilatus -2, cast by acquisition
    result=calibrate_direct_beam(angles,images,detector='Pilatus100k',
        beam_pixel=beam,angle_zero_deg=0)
    assert result.image_shape == shape
    assert result.pixel_pitch_m == pytest.approx(172e-6)
    assert result.distance_m == pytest.approx(.5,rel=1e-4)
    assert not result.used[2]


@pytest.mark.parametrize('mask_matches_image', [False, True])
def test_registered_binning_preserves_configured_mask(mask_matches_image):
    from pyFAI.detectors import detector_factory
    angles = np.array([1., .5, 0., -.5, -1.])
    shape = (65, 487)
    images, _, beam = beam_images(angles, shape=shape,
        pitches=(516e-6, 172e-6), distance=.5, zero=0.)
    detector = detector_factory('Pilatus100k')
    if mask_matches_image:
        assert detector.guess_binning(shape)
    mask = np.zeros(shape if mask_matches_image else detector.shape, np.int8)
    mask[:, 243] = 1
    detector.mask = mask

    if mask_matches_image:
        result = calibrate_direct_beam(angles, images, detector=detector,
            beam_pixel=beam, angle_zero_deg=0.)
        np.testing.assert_array_equal(result.used, [True, True, False, True, True])
        assert 'masked/gap' in result.rejection_reasons[2]
        assert result.distance_m == pytest.approx(.5, rel=2e-5)
    else:
        with pytest.raises(ValueError, match='detector mask shape differs'):
            calibrate_direct_beam(angles, images, detector=detector,
                beam_pixel=beam, angle_zero_deg=0.)

    assert detector.shape == (shape if mask_matches_image else (195, 487))
    assert detector.binning == ((3, 1) if mask_matches_image else (1, 1))
    np.testing.assert_array_equal(detector.mask, mask)


def test_wide_scan_and_positive_statistical_uncertainty():
    angles=np.array([12.,16.,22.,23.,28.,34.,37.])
    images,centres,beam=beam_images(angles,shape=(41,481),distance=.04,zero=23.)
    rng=np.random.default_rng(328)
    for im in images: im += rng.normal(0,.3,im.shape)
    result=calibrate_direct_beam(angles,images,detector='Detector',
        detector_config=config((41,481)),beam_pixel=(beam[0]+15,beam[1]+1),angle_zero_deg=23.)
    assert result.distance_m == pytest.approx(.04,rel=2e-5)
    assert 0 < result.distance_std_m < 1e-6
    np.testing.assert_allclose(result.centres_px,centres,atol=.001)


def test_custom_corners_not_hidden_by_detector_clone():
    from pyFAI.detectors import Detector
    det=Detector(pixel1=120e-6,pixel2=80e-6,max_shape=(45,161))
    det.set_pixel_corners(det.get_pixel_corners().copy())
    with pytest.raises(ValueError,match='uniform, flat'):
        calibrate_direct_beam([0,1,2],[],detector=det,beam_pixel=(80,22),angle_zero_deg=0)


def test_constant_peak_is_degenerate_and_extra_frame_is_error():
    angles=[6,7,8]
    images,_,_=beam_images([7,7,7])
    with pytest.raises(ValueError,match='degenerate'):
        run(angles,images)
    with pytest.raises(ValueError,match='length'):
        run(angles,images+images)


def test_streaming_does_not_retain_input_frames():
    import weakref
    angles=[8.,7.5,7.,6.5,6.]
    previous=[]
    def stream():
        for angle in angles:
            assert not previous or previous[-1]() is None
            image=beam_images([angle])[0][0]
            previous.append(weakref.ref(image))
            yield image
            del image
    ids = tuple(('run-uid', event) for event in (18, 3, 25, 4, 6))
    result=run(angles,stream(),frame_ids=ids,point_ids=ids)
    assert result.used.all()
    assert result.frame_ids == result.point_ids == ids
    assert all(ref() is None for ref in previous)
