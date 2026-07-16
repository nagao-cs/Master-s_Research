from src.boundingBox.boundingBox import DetectionBoundingBox

def calc_jaccard(bbox_groups: list[list[DetectionBoundingBox]]) -> float:
        if not bbox_groups:
            return 1.0
        agreed = sum(1 for g in bbox_groups if len(g) == 2)
        return agreed / len(bbox_groups)