"""Build the 29-actuator DR02 model with fixed neck and legs/waist/arms order."""
from pathlib import Path
import tempfile
import os
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE_XML = ROOT / 'mujoco/dr02_pro/upstream/DR02/mjcf/pro/DR02_pro.xml'
JOINT_NAMES = [
    f'{side}_{joint}_joint'
    for side in ('left', 'right')
    for joint in ('hip_y', 'hip_x', 'hip_z', 'knee', 'ankle_y', 'ankle_x')
] + ['waist_z_joint', 'waist_x_joint', 'waist_y_joint'] + [
    f'{side}_{joint}_joint'
    for side in ('left', 'right')
    for joint in ('shoulder_y', 'shoulder_x', 'shoulder_z', 'elbow', 'wrist_z', 'wrist_y', 'wrist_x')
]


def build_model(destination, *, ground=False):
    """Keep the source geometry/inertias, fixing only neck_z and neck_y at zero."""
    tree = ET.parse(SOURCE_XML)
    root = tree.getroot()
    root.set('model', 'DR02_pro_29dof_umr')
    root.find('compiler').set('meshdir', str((SOURCE_XML.parent / 'meshes').resolve()))
    fixed = {'neck_z_joint', 'neck_y_joint'}
    removed = []
    for parent in root.iter():
        for child in list(parent):
            if child.tag == 'joint' and child.get('name') in fixed:
                removed.append(child.get('name'))
                parent.remove(child)
    if set(removed) != fixed:
        raise ValueError('Source XML no longer has the expected two neck joints')
    base = root.find("worldbody/body[@name='base_link']")
    free = base.find('freejoint')
    if free is None:
        free = base.find("joint[@type='free']")
    free.set('name', 'floating_base_joint')
    branches = {body.get('name'): body for body in base.findall('body')}
    order = ('left_hip_y_link', 'right_hip_y_link', 'waist_z_link')
    if set(branches) != set(order):
        raise ValueError('Unexpected source root branches')
    for body in branches.values():
        base.remove(body)
    for name in order:
        base.append(branches[name])
    actuators = root.find('actuator')
    motors = {motor.get('joint'): motor for motor in actuators}
    if set(motors) != set(JOINT_NAMES) | fixed:
        raise ValueError('Unexpected source actuator joints')
    for motor in list(actuators):
        actuators.remove(motor)
    for name in JOINT_NAMES:
        actuators.append(motors[name])
    if ground:
        ET.SubElement(root.find('worldbody'), 'geom', name='bones_umr_ground', type='plane',
                      pos='0 0 0', size='100 100 0.1', contype='1', conaffinity='1')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space='  ')
    content = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    if destination.exists() and destination.read_bytes() == content:
        return destination
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.robot-', suffix='.xml', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    os.replace(temporary, destination)
    return destination


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ground', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output path; existing assets are never overwritten')
    print(build_model(args.output, ground=args.ground))
