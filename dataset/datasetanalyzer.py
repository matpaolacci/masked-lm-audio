import yaml, os, pickle
import matplotlib.pyplot as plt
from collections import Counter
from enum import Enum
from math import ceil
from random import sample
from zipfile import ZipFile
from tempfile import NamedTemporaryFile
from tqdm import tqdm

DATASET_DIR = None
WORKING_DIR = None

"""
List of all instruments used in the tracks provided by the dataset
[
    'Bass', 'Brass', 'Chromatic Percussion', 'Drums', 'Ethnic', 'Guitar', 'Organ', 'Percussive', 
    'Piano', 'Pipe', 'Reed', 'Sound Effects', 'Sound effects', 'Strings', 'Strings (continued)', 
    'Synth Lead', 'Synth Pad'
]
"""

############# TYPES & EXCEPTIONS ##############

class InstrumentType(Enum):
    BASS = 'Bass'
    BRASS = 'Brass'
    CHROMATIC_PERCUSSION = 'Chromatic Percussion'
    DRUMS = 'Drums'
    ETHNIC = 'Ethnic'
    GUITAR = 'Guitar'
    ORGAN = 'Organ'
    PERCUSSIVE = 'Percussive'
    PIANO = 'Piano'
    PIPE = 'Pipe'
    REED = 'Reed'
    SOUND_EFFECTS = 'Sound Effects'
    STRINGS = 'Strings'
    STRINGS_CONTINUED = 'Strings (continued)'
    SYNTH_LEAD = 'Synth Lead'
    SYNTH_PAD = 'Synth Pad'

class SetType(Enum):
    """Describe the set used to train the model.
    """
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

class ValueIsNotAPercentageException(Exception):
    def __init__(self) -> None:
        super().__init__("The provided value is not a valid percentage")

################# FUNCTIONS ################## 

def createTracksInfoFile(setType: SetType) -> None:
    tracks = dict()

    dataDir = os.path.join(DATASET_DIR, setType.name)

    # At WORK_DIR path there are only directories
    for track in sorted(os.listdir(dataDir)):
        itemPath = os.path.join(dataDir, track)
        
        if os.path.isdir(itemPath):
            # Load the YAML file
            setOfInstruments = set()
            with open(os.path.join(itemPath, "metadata.yaml"), 'r') as file:
                data = yaml.safe_load(file)
                for s in data["stems"]:
                    setOfInstruments.add(data["stems"][s]["inst_class"])
            tracks[track] = setOfInstruments

    with open(os.path.join(WORKING_DIR, setType.value + "_info.data"), 'wb') as f:
        f.write(pickle.dumps(tracks))

def loadTracksInfo(setType: SetType) -> dict:
    with open(os.path.join(WORKING_DIR, setType.value + "_info.data"), 'rb') as f:
        return pickle.loads(f.read())
    
def loadTracksInfoSubset(setType: SetType, instrumentsFilter: set = None, percentageSize: float = None) -> dict:
    """Given the set of instruments "instrumentsFiletr" this function returns the track infos of the tracks for which "instrumentsFilter
    is a subset of the instruments that compose each one track. If percentageSize is not None the returned dictionary contains the percentageSize%
    of the tracks in the original set; the choice of the tracks is casual.

    Args:
        setType (SetType)
        filterInstruments (set): the set of instruments that the we want they compose the returned tracks.
        percentageSize (float): a value in the interval (0,1] representing the percentage size of the specified 
        set that should be returned, relative to the original size.

    Returns:
        dict: tracks informations
    """
    trackInfos = loadTracksInfo(setType)
    
    if instrumentsFilter is not None:
        # Filters the tracks that are composed by the instruments specified in instrumentsFilter
        filteredTracks = {track: trackInstruments for track, trackInstruments in trackInfos.items() if instrumentsFilter.issubset(trackInstruments)}
    else:
        filteredTracks = trackInfos

    if percentageSize > 1.0 or percentageSize < 0:
        raise ValueIsNotAPercentageException()

    if percentageSize is not None:
        # List of the track names
        trackNames = list(filteredTracks.keys())

        # Select randomly a subset of tracks
        randomTracksSubset = sample(trackNames, ceil(len(trackNames) * percentageSize))

        # Build the dictionary with the selected subset of tracks
        # Costruisci il nuovo dizionario con le tracce selezionate
        filteredTracks = {track: filteredTracks[track] for track in randomTracksSubset}

    return filteredTracks

def buildInstrumentsUsageFrequencyHistogram(setType: SetType, savePath: str = None) -> None:
    infosDict = loadTracksInfo(setType)
    _buildInstrumentsUsageFrequencyHistogram(
        infosDict, 
        os.path.join(savePath, setType.value + "_histogram.png") if savePath else None, 
        setType
    )

def getListOfAllInstruments(setType: SetType) -> list:
    """Return a sorted list containing all instruments played in the tracks of the 
    selected set.

    Args:
        setType (SetType): One between train, validation or test set

    Returns:
        list: Sorted list of instruments
    """
    infosDict = loadTracksInfo(setType)
    allInstr = set()

    for strumenti in infosDict.values():
        allInstr.update(strumenti)

    return sorted(allInstr)

def buildDatasetZip(instrumentsFilter: set, percentageSize: float) -> None:
    """Build a zip containing three folders: train, validation and test. Each one of these contains 
    tracks for which "instrumentsFilter" is a subset of the instruments that compose each track.

    Arguments:
        instrumentsFilter (set): a set of instruments
        percentageSize (float): a value in (0,1] inverval representing the percentage size of each set
            (train, validation, test), with respect to original size. 
    """

    def addDirectoryToZip(sourceDirPath: str, zipFile: ZipFile, archiveDirectory: str):
        """Add the entire directory to zipFile.

        Arguments:
            sourceDirPath (str): The directory you want to add to the zipFile
            zipFile (ZipFile).
            archiveDirectory (str): the directory where all files are placed.
        """

        for root, _, files in os.walk(sourceDirPath):
            for file in files:
                if file.startswith("."):
                    continue
                filepath = os.path.join(root, file)
                relativeFilePath = os.path.relpath(filepath, sourceDirPath)
                archiveFilePath = os.path.join(archiveDirectory, relativeFilePath)
                zipFile.write(filepath, arcname=archiveFilePath)

    def addSetToZip(zipFile: ZipFile, setType: SetType, trackList: list[str]):
        """Add the track directories of the slakh2100 dataset, specified in the trackList, to the zip.

        Args:
            zipFile (ZipFile)
            setType (SetType)
            trackList (list): the list of the track you want to add to the zip
        """
        sourceDirPath = os.path.join(DATASET_DIR, setType.value)
        for trackName in tqdm(trackList, desc=f"Adding {setType.value} set"):
            addDirectoryToZip(os.path.join(sourceDirPath, trackName), zipFile, os.path.join(setType.value, trackName))
        print(f"{setType.value} set was created in the zip file!")

    # Get track informations dictionaries
    infoTrain = loadTracksInfoSubset(SetType.TRAIN, instrumentsFilter, percentageSize)
    infoTest = loadTracksInfoSubset(SetType.TEST, instrumentsFilter, percentageSize)
    infoValidation = loadTracksInfoSubset(SetType.VALIDATION, instrumentsFilter, percentageSize)

    # path of the zip file to create
    zipFilePath = os.path.join(WORKING_DIR, "dataset.zip")

    # Build the zip file
    with ZipFile(zipFilePath, 'w') as zipFile:
        addSetToZip(zipFile, SetType.TRAIN, list(infoTrain.keys()))
        addSetToZip(zipFile, SetType.VALIDATION, list(infoValidation.keys()))
        addSetToZip(zipFile, SetType.TEST, list(infoTest.keys()))
        
        # Add the histograms
        for setType, trackInfo in { SetType.TRAIN: infoTrain, SetType.TEST: infoTest, SetType.VALIDATION: infoValidation }.items():
            with NamedTemporaryFile(delete=False, suffix=".png") as tempFile:
                _buildInstrumentsUsageFrequencyHistogram(trackInfo, tempFile.name, setType)
                tempFile.close()
                zipFile.write(tempFile.name, arcname= os.path.join("histograms", setType.value + ".png"))
                os.remove(tempFile.name)


################## UTILITIES #################

def loadConfigPathsYaml(pathToConfigPathsYaml: str) -> None:
    global WORKING_DIR
    global DATASET_DIR

    with open(pathToConfigPathsYaml, 'r') as yamlFile:
        constants = yaml.safe_load(yamlFile)

        WORKING_DIR = constants["working_dir"]
        DATASET_DIR = constants["dataset_dir"]
    
def _buildInstrumentsUsageFrequencyHistogram(info: dict, savePath: str = None, setLabel: SetType = None) -> None:
    """Build the instrument frequencies histogram and saves it in the file specified by the 'savePath' path, if savePath is provided;
    Else it will only show it.

    Args:
        info (dict): the dictionary containing the informations about the tracks
        savePath (str, optional): If provided the histogram will be saved and will not be shown, else it will only be shown. Defaults to None.
    """
    
    # Counts the instruments frequency
    instruments = [strumento for instruments in info.values() for strumento in instruments]
    instrumentsFrequency = Counter(instruments)

    # Extract instruments and frequencies
    instruments = list(instrumentsFrequency.keys())
    frequencies = list(instrumentsFrequency.values())
 
    # Create histogram and show it
    plt.figure(figsize=(10, 6))
    plt.bar(instruments, frequencies, color='skyblue')
    plt.xlabel('Instruments')
    plt.ylabel('Frequence')
    plt.title(f"Frequencies of the instruments in the {setLabel.value if setLabel != None else 'selected'} set")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if not savePath:
        plt.show()
    else:
        plt.savefig(savePath)
