from pathlib import Path
from src.boundingBox.boundingBox import DetectionBoundingBox

class FileWriter:
    def __init__(self, output_dir: Path):
        self.output_dir: Path = output_dir
        self.output_dir.mkdir(mode=511, parents=True, exist_ok=True)
        
    def write(self, file_name: str, detection_list: list[DetectionBoundingBox]) -> bool:
        """
        write detection result to desined path
        Args :
            file_name: ファイル名
            detection_list: 検出結果
        Return :
            保存の正否
        """
        output_path: Path = self.output_dir / file_name
        if output_path.exists(): # 存在するファイル名か確認
            raise FileExistsError("すでに存在するファイル名です")
            return False
        with open(output_path, mode='w') as output_file:
            for bbox in detection_list:
                output_file.write(
                    f"{bbox.classId} {bbox.xCenter} {bbox.yCenter} "
                    f"{bbox.width} {bbox.height} {bbox.confidenceScore}\n"
                )
        
        return True