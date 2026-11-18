"""Patient-coordinate axial reorientation using DICOM IOP, never vendor labels.

Canonical columns increase towards patient left (LPS +X); rows towards
posterior (+Y). Only lossless transpose/flips are permitted. Oblique/non-axial
or missing geometry is rejected, not silently resliced or inferred from vendor.
"""
import numpy as np


def orientation_plan(iop,shape=(128,128),tolerance=.02):
    iop=np.asarray(iop,dtype=float)
    if iop.shape!=(6,) or not np.isfinite(iop).all():raise ValueError('Missing/invalid ImageOrientationPatient')
    if len(shape)!=2 or min(shape)<1:raise ValueError('Invalid image grid')
    # DICOM first triplet is along increasing COLUMN index, second along ROW.
    column,row=iop[:3],iop[3:]
    if abs(np.linalg.norm(column)-1)>.001 or abs(np.linalg.norm(row)-1)>.001 or abs(column@row)>.001:
        raise ValueError('IOP must contain orthonormal unit directions')
    vectors=[row,column]
    dominant=[int(np.argmax(np.abs(v))) for v in vectors]
    if set(dominant)!={0,1}:raise ValueError('Only near-axial biped geometry is supported')
    for v,axis in zip(vectors,dominant):
        remainder=np.delete(v,axis)
        if np.linalg.norm(remainder)>tolerance:raise ValueError('Oblique geometry requires a separate reslicing protocol')
    order=[dominant.index(1),dominant.index(0)]
    flips=[bool(vectors[order[0]][1]<0),bool(vectors[order[1]][0]<0)]
    canonical_shape=[int(shape[order[0]]),int(shape[order[1]])]
    transform=np.zeros((3,3),float);transform[2,2]=1
    for canonical_axis,original_axis in enumerate(order):
        transform[original_axis,canonical_axis]=-1 if flips[canonical_axis] else 1
        transform[original_axis,2]=canonical_shape[canonical_axis]-1 if flips[canonical_axis] else 0
    return dict(version=1,transpose=order==[1,0],flip_rows=flips[0],flip_columns=flips[1],
                source_shape=list(map(int,shape)),canonical_shape=canonical_shape,
                canonical_to_source_index=transform.tolist(),alignment_tolerance=tolerance,
                target='columns +LPS X; rows +LPS Y; no oblique reslicing')


def apply_orientation(array,plan):
    array=np.asarray(array)
    if list(array.shape)!=plan['source_shape']:raise ValueError('Array must match source grid')
    result=array.T if plan['transpose'] else array
    if plan['flip_rows']:result=np.flip(result,axis=0)
    if plan['flip_columns']:result=np.flip(result,axis=1)
    return np.ascontiguousarray(result)


def invert_orientation(array,plan):
    array=np.asarray(array)
    if list(array.shape)!=plan['canonical_shape']:raise ValueError('Array must match canonical grid')
    result=array
    if plan['flip_columns']:result=np.flip(result,axis=1)
    if plan['flip_rows']:result=np.flip(result,axis=0)
    if plan['transpose']:result=result.T
    return np.ascontiguousarray(result)


def resized_pixel_map(iop,position,pixel_spacing,native_shape,processed_shape,plan):
    """3x3 map [row,col,1] -> LPS millimetres at resized pixel centres.

    Assumes the documented whole-image resize with no crop/rotation. It does
    not pretend nominal slice thickness is a verified inter-slice spacing.
    """
    iop=np.asarray(iop,float);position=np.asarray(position,float);spacing=np.asarray(pixel_spacing,float)
    if position.shape!=(3,) or spacing.shape!=(2,) or not np.isfinite(position).all() or not np.isfinite(spacing).all() or (spacing<=0).any():
        raise ValueError('Invalid position or pixel spacing')
    factors=np.asarray(native_shape,float)/np.asarray(processed_shape,float)
    if not np.isfinite(factors).all() or (factors<=0).any():raise ValueError('Invalid native/processed shape')
    row_step=iop[3:]*spacing[0]*factors[0];column_step=iop[:3]*spacing[1]*factors[1]
    origin=position+iop[3:]*spacing[0]*(factors[0]/2-.5)+iop[:3]*spacing[1]*(factors[1]/2-.5)
    source=np.column_stack([row_step,column_step,origin])
    canonical=source@np.asarray(plan['canonical_to_source_index'])
    return dict(source_pixel_to_lps=source.tolist(),canonical_pixel_to_lps=canonical.tolist(),
                canonical_row_spacing_mm=float(np.linalg.norm(canonical[:,0])),
                canonical_column_spacing_mm=float(np.linalg.norm(canonical[:,1])),
                resize_assumption='whole image, centre-aligned resize; no crop; verified on audit samples')


def transform_landmarks(points,plan,to_source=False):
    matrix=np.asarray(plan['canonical_to_source_index'])
    if not to_source:matrix=np.linalg.inv(matrix)
    result={}
    for name,xy in points.items():
        xy=np.asarray(xy,float)
        if xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError('Landmarks must be finite [x,y]')
        rc=matrix@np.array([xy[1],xy[0],1.]);result[name]=[float(rc[1]),float(rc[0])]
    return result
