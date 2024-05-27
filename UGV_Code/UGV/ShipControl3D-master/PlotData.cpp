/*****************************************************************
**				Project:	ShipControl(WOPC)					**
**				Author:		Dong Shengwei						**
**				Library:	BestSea								**
**				Date:		2014-01-07						**
******************************************************************/

//PlotData.cpp

#include "PlotData.h"
#include <QDebug>

PlotData::PlotData(void)
{
	init();
}


PlotData::~PlotData(void)
{
	engClose(ep);
}

//��ʼ��
void PlotData::init()
{
	ep = NULL;
	if (!(ep=engOpen("")))
	{
		qDebug() << "ep is opened fail!" << endl;
		exit(EXIT_FAILURE);
	}
}

void PlotData::drawCurve()
{
	engEvalString(ep, "cd E:/projectProgram/data/ ");	
	engEvalString(ep, "close all");	
	engEvalString(ep, "plotData");	
}
