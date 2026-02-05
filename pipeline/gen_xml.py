import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.dom import minidom
import spikeinterface.extractors as se

def cmap(shank_id):
    shank_id = int(shank_id)
    if shank_id == 0:
        return "#5081eb"
    elif shank_id == 1:
        return '#00ff7f'
    elif shank_id == 2:
        return '#e59900'
    else:
        return '#ff3898'

def create_neuroscope_xml(probe_map, fs = 30000, n_channels = 384, pad_groups=0, sync_channel=False) -> str:    
    # Shank 0: 1-50 (um) (x coordinates), Shank 1: 51-100, Shank 2: 101-150, Shank 3: 151-200
    # Sort by shank_id ascending and yc descending
    probe_map.sort_values(['shank_ids', 'y'], ascending=[True, False], inplace=True)
    # Map colors for shanks
    probe_map['cmap'] = probe_map['shank_ids'].apply(cmap)

    # Root element
    root = ET.Element("parameters")
    root.set('version', "1.0")
    root.set('creator', "neuroscope-2.0.0")

    # Acquisition System
    # field Potentials
    # Anatomical Description
    # Spike Detection
    # NeuroScope

    acq                 = ET.SubElement(root, "acquisitionSystem")
    fieldPotentials     = ET.SubElement(root, "fieldPotentials")
    anat                = ET.SubElement(root, "anatomicalDescription")
    spikeDetection      = ET.SubElement(root, "spikeDetection")
    neuroscope          = ET.SubElement(root, "neuroscope")
    
    ET.SubElement(acq, "nBits").text = "16" 
    if sync_channel:
        ET.SubElement(acq, "nChannels").text = str(n_channels + 1)
    else:
        ET.SubElement(acq, "nChannels").text = str(n_channels)
    ET.SubElement(acq, "samplingRate").text = str(fs)
    ET.SubElement(acq, "voltageRange").text = "20"
    ET.SubElement(acq, "amplification").text = "1000"
    ET.SubElement(acq, "offset").text = "0"

    ET.SubElement(fieldPotentials, "lfpSamplingRate").text = str(fs)

    groups = ET.SubElement(anat, "channelGroups")
    # Create channel groups based on shank IDs
    shanks = probe_map['shank_ids'].unique()
    for i, shank in enumerate(shanks):
        group = ET.SubElement(groups, 'group')
        shank_channels = probe_map[probe_map['shank_ids'] == shank]
        # Add channels to group (sorted by depth (y) descending)
        for j, row in shank_channels.iterrows():
            channel = ET.SubElement(group, 'channel')
            channel.set('skip', '0')
            channel.text = str(j)
        # Use the channel 0 as skip channels to separate groups on neuroscope
        if pad_groups > 0 and i < len(shanks) - 1:
            for _ in range(pad_groups):
                spacer = ET.SubElement(group, 'channel')
                spacer.set('skip', '1')
                # Use channel 0 as skip channel
                spacer.text = str(shank_channels.index[0])
    
    if sync_channel:
        sync = ET.SubElement(groups, 'group')
        spacer = ET.SubElement(sync, 'channel')
        spacer.set('skip', '1')
        spacer.text = str(n_channels)

    neuroscope.set('version', "2.0.0")

    miscellaneous = ET.SubElement(root, "miscellaneous")
    ET.SubElement(miscellaneous, "screenGain").text = "0.5"
    ET.SubElement(miscellaneous, "traceBackgroundImage")

    video = ET.SubElement(neuroscope, "video")
    ET.SubElement(video, "rotate").text = "0"
    ET.SubElement(video, "flip").text = "0"
    ET.SubElement(video, 'videoImage')
    ET.SubElement(neuroscope, "positionsBackground").text = "0"

    spikes = ET.SubElement(neuroscope, "spikes")
    ET.SubElement(spikes, "nSamples").text = "32"
    ET.SubElement(spikes, "peakSampleIndex").text = "16"

    channels = ET.SubElement(neuroscope, 'channels')
    for i, row in probe_map.sort_index().iterrows():
        chan_id = str(i)
        color = row['cmap']
        
        colors = ET.SubElement(channels, 'channelColors')
        ET.SubElement(colors, 'channel').text = chan_id
        ET.SubElement(colors, 'color').text = color
        ET.SubElement(colors, 'anatomyColor').text = color
        ET.SubElement(colors, 'spikeColor').text = color
        
        offset = ET.SubElement(channels, 'channelOffset')
        ET.SubElement(offset, 'channel').text = chan_id
        ET.SubElement(offset, 'defaultOffset').text = "0"

    return minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")


def main():
    parser = argparse.ArgumentParser(description='Generate Neuroscope XML from OpenEphys recording.')
    parser.add_argument('session_path', type=str, help='Path to the OpenEphys session directory')
    parser.add_argument('--pad_groups', type=int, default=10, help='Number of skip channels to pad between groups (default: 10)')
    parser.add_argument('--fs', type=int, default=30000, help='Sampling frequency (default: 30000)')
    args = parser.parse_args()
    
    # Load recordings with SpikeInterface
    stream_names, stream_ids = se.get_neo_streams('openephysbinary', args.session_path)

    
    # Check if there is extra sync channel (385th channel)
    SYNC = False
    for stream_name in stream_names:
        if 'SYNC' in stream_name:
            SYNC = True
            break
    
    for stream_name, stream_id in zip(stream_names, stream_ids):
        print(f"Found stream: {stream_name} ({stream_id})")
        if 'Probe' in stream_name and 'SYNC' not in stream_name:
            print(f"Loading stream: {stream_name}({stream_id})")
            rec = se.read_openephys(args.session_path, stream_id=stream_id)
            
            # Get actual recording parameters
            fs = int(rec.get_sampling_frequency())
            n_channels = rec.get_num_channels()

            for stream_folder in rec._stream_folders:
                # Path object
                fname = stream_folder / "continuous.xml"
                if fname.exists():
                    print(f"XML configuration already exists: {fname}")
                else: 
                    print(f"Writing XML configuration to: {fname}")
                    xml_str = create_neuroscope_xml(rec.get_probe().to_dataframe(),
                                                    fs=fs,
                                                    n_channels=n_channels,
                                                    pad_groups=args.pad_groups,
                                                    sync_channel=SYNC)
                    with open(fname, "w") as f:
                        f.write(xml_str)

if __name__ == '__main__':
    main()