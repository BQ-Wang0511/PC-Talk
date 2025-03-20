window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation_video";
// var NUM_INTERP_FRAMES = 100;

function preloadInterpolationVideos(prefix) {
	var videos = [];
	if (prefix.startsWith('scale')){
		NUM_INTERP_FRAMES = 60;
	}
	else{
		NUM_INTERP_FRAMES=100;
	}

	for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
		var path = INTERP_BASE + '/' +prefix+'/'+ String(i).padStart(3, '0') + '.mp4';
		var video = document.createElement('video');
		video.src = path;
		video.type = 'video/mp4';
		video.controls = true;
		videos[i] = video;
		// if(prefix.startsWith('scale')){
		// 	video.volume=30+i*0.2
		// }
	}
	return videos;
}

function setInterpolationVideo(i, videos, wrapperSelector) {
	var video = videos[i];
	video.ondragstart = function() { return false; };
	video.oncontextmenu = function() { return false; };
	$(wrapperSelector).empty().append(video);
	var endPrefixParts = wrapperSelector.split('-');
	var endPrefix = endPrefixParts[endPrefixParts.length - 1]; // 获取数组的最后一个元素

	if (!endPrefix.startsWith('emo')) {
		video.volume = 0.1 + i * 0.015;
	}
}

$(document).ready(function() {
	var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
	}

	// Initialize all div with carousel class
	var carousels = bulmaCarousel.attach('.carousel', options);

	var videos1 = preloadInterpolationVideos('scale1');
	$('#interpolation-slider-1').on('input', function() {
		setInterpolationVideo(this.value, videos1, '#interpolation-video-wrapper-1');
	});
	setInterpolationVideo(0, videos1, '#interpolation-video-wrapper-1');

	var videos2 = preloadInterpolationVideos('scale2');
	$('#interpolation-slider-2').on('input', function() {
		setInterpolationVideo(this.value, videos2, '#interpolation-video-wrapper-2');
	});
	setInterpolationVideo(0, videos2, '#interpolation-video-wrapper-2');

	var videos_emo1 = preloadInterpolationVideos('intense1');
	$('#interpolation-slider-emo1').on('input', function() {
		setInterpolationVideo(this.value, videos_emo1, '#interpolation-video-wrapper-emo1');
	});
	setInterpolationVideo(0, videos_emo1, '#interpolation-video-wrapper-emo1');

	var videos_emo2 = preloadInterpolationVideos('intense2');
	$('#interpolation-slider-emo2').on('input', function() {
		setInterpolationVideo(this.value, videos_emo2, '#interpolation-video-wrapper-emo2');
	});
	setInterpolationVideo(0, videos_emo2, '#interpolation-video-wrapper-emo2');

	// var videos_tmp = preloadInterpolationVideos('tmp1');
	// $('#interpolation-slider-tmp').on('input', function() {
	// 	setInterpolationVideo(this.value, videos_tmp, '#interpolation-video-wrapper-tmp');
	// });
	// setInterpolationVideo(0, videos_tmp, '#interpolation-video-wrapper-tmp');

	bulmaSlider.attach();
});

