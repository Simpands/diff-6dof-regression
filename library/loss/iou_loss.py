

def iou_loss(im1, im2):
    intersection = im1 * im2
    union = im1 + im2 - intersection

    intersection = intersection.sum((1, 2))
    union = union.sum((1, 2))

    iou = intersection / union

    return 1. - iou.mean()
