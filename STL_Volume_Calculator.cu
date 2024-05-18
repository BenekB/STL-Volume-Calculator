//	author: Benedykt Bela

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>



//	funkcja na GPU sumuj¹ca ze sob¹ dwa wektory danych 
__global__ void add(float* R_side, float* L_side, int size,  float accuracy)
{
    int i = threadIdx.x;

	R_side[i] = R_side[i] - L_side[i];
	R_side[i] = R_side[i] * accuracy * accuracy;
}



//	funkcja na GPU s³u¿¹ca do podzielenia odcinka na punkty zgodnie z zadan¹ dok³adnoœci¹
//	dla kolejnych wartoœci zmiennej y punkt przeciêcia prostej y i odcinka mo¿emy szukaæ naraz
__global__ void crossline_cuda(float* y0, float* y1, float* x0, float* x1, char* strona, float* y)
{
	struct point2D		
	{
		float x;
		float y;
	};


	struct direct_point
	{
		point2D point;
		char strona;
	};


	//	tworzê nowy punkt przeciêcia odcinka z prost¹ y
	direct_point* punkt = new direct_point;
	point2D pomocniczy;		//	dodatkowy punkt usprawnia obliczenia


	int i = threadIdx.x;

	//	je¿eli odcinek koñczy siê i zaczyna w tym samym punkcie
	if (y0[i] == y1[i] && x0[i] == x1[i])
		punkt[0].strona = 'P';
	else
	{
		//	szukanie przeciêcia prostej y z zadanym odcinkiem
		pomocniczy.x = (x0[i] - x1[i]) * (y[i] - y1[i]) / (y1[i] - y0[i]);
		punkt[0].point.x = x1[i] + pomocniczy.x;
		punkt[0].strona = strona[i];
		punkt[0].point.y = y[i];
	}


	//	je¿eli znaleziony punkt jest punktem skrajnym odcinka
	if (y0[i] < y1[i] && y1[i] == punkt[0].point.y)
		punkt[0].strona = 'P';
	if (y0[i] > y1[i] && y0[i] == punkt[0].point.y)
		punkt[0].strona = 'P';


	//	do wektora, który skopiujê na CPU zapisujê dane znalezionego punktu preciêcia
	y0[i] = punkt[0].point.y;
	x0[i] = punkt[0].point.x;
	strona[i] = punkt[0].strona;

}



//	funkcja na GPU s³u¿¹ca do podzia³u trójk¹ta na odcinki poprzec robienie przekroju 
//	przesuwaj¹c¹ siê p³aszczyzn¹
__global__ void crossection_cuda(float* y0, float* y1, float* y2, float* x0, 
								float* x1, float* x2, float* z0, 
								float* z1, float* z2, char* strona, float* z, float* normx)
{
	struct point3D
	{
		float x;
		float y;
		float z;
	};


	struct point2D
	{
		float x;
		float y;
	};


	struct line
	{
		point2D point[2];
		char strona;
	};


	//	tworzê struktury linia oraz punkt w przestrzeni trójwymiarowej, 
	//	które bêd¹ pomocne przy dalszych obliczeniach
	line* linia = new line();
	point3D pomocniczyp;

	int j = 0;		//	potrzebne do zliczania który punkt aktualnie zapisujê do struktury linia
	int i = threadIdx.x;

	
	//	poni¿sze instrukcje warunkowe sprawdzaj¹ miêdzy którymi punktami znajduje siê obecnie 
	//	przeszukiwana p³aszczyzna, czyli które odcinki trójk¹ta bêdziemy przecinaæ oraz te odcinki
	//	przecina
	if ((z1[i] >= z[i] > z0[i] || z1[i] < z[i] <= z0[i]) && ((z0[i] - z1[i]) != 0))
	{
		pomocniczyp.x = x0[i] - x1[i];
		pomocniczyp.y = y0[i] - y1[i];
		pomocniczyp.x = pomocniczyp.x * (z[i] - z1[i]) / (z0[i] - z1[i]);
		pomocniczyp.y = pomocniczyp.y * (z[i] - z1[i]) / (z0[i] - z1[i]);
		linia[0].point[j].x = x1[i] + pomocniczyp.x;
		linia[0].point[j].y = y1[i] + pomocniczyp.y;

		j++;
	}

	if ((z1[i] >= z[i] > z2[i] || z1[i] < z[i] <= z2[i]) && (z2[i] - z1[i]) != 0)
	{
		pomocniczyp.x = x2[i] - x1[i];
		pomocniczyp.y = y2[i] - y1[i];
		pomocniczyp.x = pomocniczyp.x * (z[i] - z1[i]) / (z2[i] - z1[i]);
		pomocniczyp.y = pomocniczyp.y * (z[i] - z1[i]) / (z2[i] - z1[i]);
		linia[0].point[j].x = x1[i] + pomocniczyp.x;
		linia[0].point[j].y = y1[i] + pomocniczyp.y;

		j++;
	}

	if ((z0[i] >= z[i] > z2[i] || z0[i] < z[i] <= z2[i]) && (z2[i] - z0[i]) != 0)
	{
		pomocniczyp.x = x2[i] - x0[i];
		pomocniczyp.y = y2[i] - y0[i];
		pomocniczyp.x = pomocniczyp.x * (z[i] - z0[i]) / (z2[i] - z0[i]);
		pomocniczyp.y = pomocniczyp.y * (z[i] - z0[i]) / (z2[i] - z0[i]);
		linia[0].point[j].x = x0[i] + pomocniczyp.x;
		linia[0].point[j].y = y0[i] + pomocniczyp.y;

		j++;
	}


	//	sprawdzam w któr¹ stronê jest skierowana normalna danego trójk¹ta, ¿eby wiedzieæ
	//	gdzie jest œrodek badanego obiektu, a gdzie strona zewnêtrzna
	if (normx[i] > 0)
		linia[0].strona = 'R';		//	R - z prawej
	else if (normx[i] < 0)
		linia[0].strona = 'L';		//	L - z lewej
	else
		linia[0].strona = 'T';		//	T oznacza trójk¹t prostopad³y do osi y


	//	zapisujê znalezione dane do wektorów, które skopiujê na CPU
	strona[i] = linia[0].strona;
	x0[i] = linia[0].point[0].x;
	x1[i] = linia[0].point[1].x;
	y0[i] = linia[0].point[0].y;
	y1[i] = linia[0].point[1].y;

}



using namespace std;


struct point3D;
struct point2D;
struct triangles;
struct line;
struct plane2D;
class Wczytywanie;
class Szukaj;


struct point3D
{
	float x;
	float y;
	float z;
};


struct point2D
{
	float x;
	float y;
};


//	ka¿dy trójk¹t sk³ada siê z trzech punktów oraz wektora normalnego okreœlaj¹cego
//	kierunek wnêtrza figury
struct triangles
{
	point3D normalna;
	point3D first;
	point3D second;
	point3D third;
};


struct line
{
	point2D point[2];
	char strona;		//	strona oznacza kierunek wnêtrza figury, której fragmentem
						//	jest dany odcinek
};


struct direct_point
{
	point2D point;
	char strona;
};


//	plane2D zawiera jedynie wektor sk³adaj¹cy siê ze struktur linia
//	ta struktura umo¿liwia zrobienie swego rodzaju tablicy wektorów
struct plane2D
{
	vector<line> plane;
};


//	tablica wektorów zawieraj¹ca ostateczne punkty do obliczania objêtoœci bry³y
struct final_points
{
	vector<direct_point> punkt;
};



//	klasa s³u¿¹ca do wczytywania danych z pliku oraz przechowywania takich informacji
//	jak globalne minimum oraz maksimum w ka¿dej p³aszczyŸnie, iloœæ pobranych 
//	trójk¹tów, dok³adnoœæ oblciczeñ oraz funkcjê do wypisywania posiadanych danych
class Wczytywanie {
public:
	Wczytywanie();

	int ile = 0;
	float accuracy = 0.1;
	float min_z = 0;
	float min_x = 0;
	float min_y = 0;
	float max_y = 0;
	float max_z = 0;
	float max_x = 0;
	triangles *triangle;
	void wypisz_dane();


private:

};



//	g³ówna klasa programu zawieraj¹ca odpowiednie funkcje obliczaj¹ce objêtoœæ figury
class Szukaj {
public:
	Szukaj();

	//	wewn¹trz g³ównej klasy programu inicjujê instancjê klasy Wczytywanie, aby mieæ 
	//	dostêp do jej danych 
	Wczytywanie wczytywanie;
	void cut_all_triangles();
	void cut_all_lines();
	void cut_triangle(triangles triangle);
	void cut_line(line linia, float g_min, float g_max);
	void wypisz_plane();
	float objetosc = 0;		//	zmienna zawieraj¹ca wartoœæ objêtoœci naszego obiektu


private:
	int ile_warstw;		//	liczba warstw przeciêciu obiektu przesuwaj¹c¹ siê p³aszczyzn¹ z
	float global_min = wczytywanie.min_z;		//	globalne minimum obiektu w osi z
	//	generalnie zmienne wykorzstywane w innych miejscach programu
	plane2D* plane;		
	point3D* results;
	float accuracy = wczytywanie.accuracy;
	int plane_size;
	final_points* final_point;

};



//	g³ówna funkcja programu
int main()
{
	
	Szukaj szukaj;		//	inicjalizujê instancjê klasy Szukaj
	szukaj.cut_all_triangles();		//	wykonujê pociêcie wszystkich trójk¹tów
	szukaj.wypisz_plane();		//	wypisuje odcinki, które sk³adaj¹ siê na przekroje obiektu
	szukaj.cut_all_lines();		//	przecina wszystkie dostepne linie i znajduje punkty przeciêcia

	cout << endl << szukaj.objetosc << endl << endl;		//	wypisujê znalezion¹ objêtoœæ figury

}



//	poni¿sza funkcja przecina wszystkie linie tworz¹ce przekroje równoleg³e do p³aszczyzny z
//	i znajduje punkty przeciêcia na podstawie których oblicza objêtoœæ bry³y
void Szukaj::cut_all_lines()
{
	float min, max;


	//	wykonujê ciêcie i liczenie dla ka¿dego przekroju równoleg³ego do p³aszczyzny z
	for (int i = 0; i < plane_size; i++)
	{
		//	ustawiam pocz¹tkow¹ wartoœæ dla zmiennych min i max
		min = plane[i].plane[0].point[0].y;
		max = min;

		//	dla ka¿dego przekroju sprawdzam jego wspó³rzêdn¹ maksymaln¹ i minimaln¹ w osi y
		for (int j = 0; j < plane[i].plane.size(); j++)
		{
			if (min > plane[i].plane[j].point[0].y)
				min = plane[i].plane[j].point[0].y;
			if (min > plane[i].plane[j].point[1].y)
				min = plane[i].plane[j].point[1].y;
			if (max < plane[i].plane[j].point[0].y)
				max = plane[i].plane[j].point[0].y;
			if (max < plane[i].plane[j].point[1].y)
				max = plane[i].plane[j].point[1].y;
		}

		//	znalezione wartoœci min i max zaokr¹glam do najbli¿szej wartoœci y, która
		//	jest wielokrotnoœci¹ dok³adnoœci
		min = min / accuracy;
		min = ceil(min);
		min = min * accuracy;

		max = max / accuracy;
		max = floor(max);
		max = max * accuracy;

		//	tworzê now¹ tablicê punktów ostatecznych 
		final_point = new final_points[int((max - min) / accuracy + 1)];

		//	ka¿d¹ liniê w danym przekroju przecinam i generujê ostateczne punkty do obliczenia objêtoœci
		for (int j = 0; j < plane[i].plane.size(); j++)
			cut_line(plane[i].plane[j], min, max);			

		int pomocnicza = int((max - min) / accuracy + 1);		//	liczba danych w tej iteracji pêtli
		
		//	wektory na CPU i GPU z wartoœciami do obliczeñ
		float* R_side = new float[pomocnicza];
		float* d_R_side = new float[pomocnicza];
		float* L_side = new float[pomocnicza];
		float* d_L_side = new float[pomocnicza];

		//	wszystkie dane przegl¹damy i zapisujemy te ze stron¹ 'R' do jedgeno wektora, 
		//	a te ze stron¹ 'L' do drugeigo wektora
		for (int k = 0; k < int((max - min) / accuracy); k++)
		{
			for (int l = 0; l < final_point[k].punkt.size(); l++)
			{
				if (final_point[k].punkt[l].strona == 'R')
					R_side[k] = final_point[k].punkt[l].point.x;

				else if (final_point[k].punkt[l].strona == 'L')
					L_side[k] = final_point[k].punkt[l].point.x;

			}

			cout << R_side[k] << " " << L_side[k] << " ";

			cout << endl;
		}

		//	alokujê miejsce na GPU
		cudaMalloc(& d_R_side, pomocnicza * sizeof(float));
		cudaMalloc(& d_L_side, pomocnicza * sizeof(float));

		//	oraz kopiujê potrzebne dane
		cudaMemcpy(d_R_side, R_side, pomocnicza * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_L_side, L_side, pomocnicza * sizeof(float), cudaMemcpyHostToDevice);

		//	wykonujê funkcjê dodawanie na GPU
		add << <1, pomocnicza >> > (d_R_side, d_L_side, pomocnicza, accuracy);

		//	i kopiujê na CPU potrzebne dane
		cudaMemcpy(R_side, d_R_side, pomocnicza * sizeof(float), cudaMemcpyDeviceToHost);

		//	sumujê skopiowane dane i dodaje do zmiennej objêtoœæ
		for (int j = 0; j < pomocnicza; j++)
			objetosc += R_side[j];

		//	usuwam dynamicznie alokowan¹ tablicê zmiennych 
		delete[] final_point;

	}
}



//	funkcja, która pobiera dane jednego obiektu line oraz globalne minimum i globalne maksimum,
//	a  zwraca punkty, które s¹ punktami przeciêcia tej linii z p³aszczyznami prostopad³ymi do osi OY
//	ustawionymi w odleg³oœci od siebie wynosz¹cej wartoœæ zmiennej accuracy
void Szukaj::cut_line(line linia, float g_min, float g_max)
{
	float min, max;

	//	ustalam który koniec linii ma wiêksz¹ wspó³rzêdn¹ y i przypisujê odpowiednio do zmiennych min i max
	if (linia.point[0].y > linia.point[1].y)
	{
		min = linia.point[1].y;
		max = linia.point[0].y;
	}
	else
	{
		min = linia.point[0].y;
		max = linia.point[1].y;
	}

	//	zaokr¹glam wartoœci min i max w tej sposób, ¿e nowe min jest najbli¿sz¹, ale wiêksz¹ wielokrotnoœci¹ 
	//	zmiennej accuracy, natomiast max jest najbli¿sz¹, ale mniejsz¹ wielokrotnoœci¹ zmiennej accuracy
	min = min / accuracy;
	min = ceil(min);
	min = min * accuracy;

	max = max / accuracy;
	max = floor(max);
	max = max * accuracy;

	//	zmienne pomocnicze s³u¿¹ce do obliczeñ
	float yorg = min;
	float y_help = yorg;
	int index = 0;
	int ile = 0;

	//	zliczam ile wielokrotnoœci zmiennej accuracy mieœci siê miêdzy wartoœciami min i max,
	//	poniewa¿ tyle punktów funkcja musi wygenerowaæ
	while (y_help <= max)
	{
		y_help += accuracy;
		ile++;
	}

	//	wektory zmiennych, do których zapisujê odpowiednie wartoœci, które póŸniej skopiujê na GPU i wykorzystam
	float* y0 = new float[ile];
	float* y1 = new float[ile];
	float* x0 = new float[ile];
	float* x1 = new float[ile];
	float* y = new float[ile];
	char* strona = new char[ile];

	//	odpowiedniki powy¿szych wskaŸników na GPU
	float* dy0 = new float[ile];
	float* dy1 = new float[ile];
	float* dx0 = new float[ile];
	float* dx1 = new float[ile];
	float* dy = new float[ile];
	char* dstrona = new char[ile];

	
	//	nadajê zainicjowanym powy¿ej wektorom odpowiednie wartoœci ze struktury linia
	for (int i = 0; i < ile; i++)
	{
		y0[i] = linia.point[0].y;
		y1[i] = linia.point[1].y;
		x0[i] = linia.point[0].x;
		x1[i] = linia.point[1].x;
		strona[i] = linia.strona;
		y[i] = yorg; 
		yorg += accuracy;
	}

	//	alokujê pamiêæ na GPU
	cudaMalloc(&dy0, ile * sizeof(float));
	cudaMalloc(&dy1, ile * sizeof(float));
	cudaMalloc(&dx0, ile * sizeof(float));
	cudaMalloc(&dx1, ile * sizeof(float));
	cudaMalloc(&dy, ile * sizeof(float));
	cudaMalloc(&dstrona, ile * sizeof(char));

	//	i kopiujê na GPU dane z CPU
	cudaMemcpy(dy0, y0, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dy1, y1, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dx0, x0, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dx1, x1, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dy, y, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dstrona, strona, ile * sizeof(char), cudaMemcpyHostToDevice);

	//	wywo³anie funkcji wykonywanej na GPU, która zosta³a opisana w miejscu definicji
	crossline_cuda << <1, ile >> > (dy0, dy1, dx0, dx1, dstrona, dy);

	//	kopiujê dane z GPU z powrotem na CPU
	cudaMemcpy(y0, dy0, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y1, dy1, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(x0, dx0, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(x1, dx1, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, dy, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(strona, dstrona, ile * sizeof(char), cudaMemcpyDeviceToHost);

	//	zwalniam pamiêæ na GPU
	cudaFree(dy0);
	cudaFree(dy1);
	cudaFree(dx0);
	cudaFree(dx1);
	cudaFree(dy);
	cudaFree(dstrona);

	//	ustawiam ponownie wartoœæ yorg na wartoœæ wczeœniej obliczonego min
	yorg = min;


	//	dla ka¿dego wyciêtego punktu dodajê go do tablicy punktów ostatecznych 
	for (int i = 0; i < ile; i++)
	{
		//	poni¿sze kilka instrukcji ustala indeks w tablicy punktów ostatecznych do którego dany punkt ma
		//	zostaæ przypisany
		float help = yorg - g_min;

		if (help < 0)
			help = 1;

		index = int(help / accuracy);

		//	tworzê nowy wskaŸnik na obiekt direct_point
		direct_point* abc = new direct_point;

		//	przypisujê do obiektu abc odpowiednie wartoœci z wektorów danych obliczonych na GPU
		abc[0].point.x = x0[i];
		abc[0].point.y = y0[i];
		abc[0].strona = strona[i];

		//	je¿eli strona danego punktu nie ma wartoœci 'P', to dodajemy ten punkt do wektora pod
		//	odpowiednim adresem tablicy final_point
		if (abc[0].strona != 'P')
			final_point[index].punkt.push_back(abc[0]);

		//	zwiêkszam aktualn¹ wartoœæ y, ¿eby nastêpny punkt zapisaæ pod innym indeksem w 
		//	tablicy final_point, bo zmienna index jest ustalana w³aœnie na podstawie yorg
		yorg += accuracy;

		//	usuwam dynamicznie alokowan¹ zmienn¹ abc
		delete abc;
	}
}



//	funkcja nale¿y do klasy Szukaj i s³u¿y do wypisania w konsoli linii tworz¹cych przekroje na poszczególnych
//	p³aszczyznach prostopad³ych do osi OZ
void Szukaj::wypisz_plane()
{
	for (int i = 0; i < ile_warstw; i++)
	{
		int k = plane[i].plane.size();

		for (int j = 0; j < k; j++)
		{
			cout << plane[i].plane[j].point[0].x << " ";
			cout << plane[i].plane[j].point[0].y << "     ";
			cout << plane[i].plane[j].point[1].x << " ";
			cout << plane[i].plane[j].point[1].y << "     ";
			cout << endl;
		}

		cout << endl;
	}
}



//	ta krótka funkcja po prostu wywo³uje funkcjê cut_triangle dla wszystkich trójk¹tów wczytanych do programu
void Szukaj::cut_all_triangles()
{
	for (int i = 0; i < wczytywanie.ile; i++)
		cut_triangle(wczytywanie.triangle[i]);

}



//	poni¿sza funkcja zwraca linie, które powstaj¹ na skutek przeciêcia zadanego trójk¹ta przez p³aszczyznê
//	OXY przesuwan¹ z krokiem accuracy wzd³u¿ osi Z
void Szukaj::cut_triangle(triangles triangle)
{
	float min, max;

	//	ustalam minimaln¹ i maksymaln¹ wspó³rzêdn¹ trójk¹ta w osi Z
	if (triangle.first.z >= triangle.second.z && triangle.first.z >= triangle.third.z)
		max = triangle.first.z;
	else if (triangle.second.z >= triangle.second.z && triangle.second.z >= triangle.third.z)
		max = triangle.second.z;
	else if (triangle.third.z >= triangle.second.z && triangle.third.z >= triangle.first.z)
		max = triangle.third.z;

	if (triangle.first.z <= triangle.second.z && triangle.first.z <= triangle.third.z)
		min = triangle.first.z;
	else if (triangle.second.z <= triangle.second.z && triangle.second.z <= triangle.third.z)
		min = triangle.second.z;
	else if (triangle.third.z <= triangle.second.z && triangle.third.z <= triangle.first.z)
		min = triangle.third.z;

	//	zaokr¹glam zmienn¹ min do najbli¿szej, wiêkszej od zmiennej min wielokrotnoœci zmiennej accuracy
	min = min / accuracy;
	min = ceil(min);
	min = min * accuracy;

	//	zaokr¹glam zmienn¹ max do najbli¿szej, mniejszej od zmiennej max wielokrotnoœci zmiennej accuracy
	max = max / accuracy;
	max = floor(max);
	max = max * accuracy;

	//	zmienne pomocnicze 
	int ile = 0;
	float zorg = min;
	float z_help = zorg;

	//	sprawdzam ile wielokrotnoœci zmiennej accuracy zmieœci siê miêdzy min i max, bo tyle linii 
	//	funkcja wytnie z trójk¹ta
	while (z_help <= max)
	{
		z_help += accuracy;
		ile++;
	}

	//	wektory zmiennych do których zapisujê odpowiednie dane, które zostan¹ skopiowane na GPU
	//	i na ich podstawie GPU dokona obliczeñ
	float* y0 = new float[ile];
	float* y1 = new float[ile];
	float* y2 = new float[ile];
	float* x0 = new float[ile];
	float* x1 = new float[ile];
	float* x2 = new float[ile];
	float* z0 = new float[ile];
	float* z1 = new float[ile];
	float* z2 = new float[ile];
	float* z = new float[ile];
	float* normx = new float[ile];
	char* strona = new char[ile];

	//	odpowiedniki powy¿szych wskaŸników wykorzystywane na GPU
	float* dy0 = new float[ile];
	float* dy1 = new float[ile];
	float* dy2 = new float[ile];
	float* dx0 = new float[ile];
	float* dx1 = new float[ile];
	float* dx2 = new float[ile];
	float* dz0 = new float[ile];
	float* dz1 = new float[ile];
	float* dz2 = new float[ile];
	float* dz = new float[ile];
	float* dnormx = new float[ile];
	char* dstrona = new char[ile];


	//	przypisujê zadeklarowanym wektorom odpowiednie dane 
	for (int i = 0; i < ile; i++)
	{
		y0[i] = triangle.first.y;
		y1[i] = triangle.second.y;
		y2[i] = triangle.third.y;
		x0[i] = triangle.first.x;
		x1[i] = triangle.second.x;
		x2[i] = triangle.third.x;
		z0[i] = triangle.first.z;
		z1[i] = triangle.second.z;
		z2[i] = triangle.third.z;
		normx[i] = triangle.normalna.x;
		z[i] = zorg;
		zorg += accuracy;
	}

	//	alokujê pamiêæ na GPU
	cudaMalloc(&dy0, ile * sizeof(float));
	cudaMalloc(&dy1, ile * sizeof(float));
	cudaMalloc(&dy2, ile * sizeof(float));
	cudaMalloc(&dz0, ile * sizeof(float));
	cudaMalloc(&dz1, ile * sizeof(float));
	cudaMalloc(&dz2, ile * sizeof(float));
	cudaMalloc(&dx0, ile * sizeof(float));
	cudaMalloc(&dx1, ile * sizeof(float));
	cudaMalloc(&dx2, ile * sizeof(float));
	cudaMalloc(&dz, ile * sizeof(float));
	cudaMalloc(&dnormx, ile * sizeof(float));
	cudaMalloc(&dstrona, ile * sizeof(char));
	
	//	i kopiujê potrzebne dane na GPU
	cudaMemcpy(dy0, y0, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dy1, y1, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dy2, y2, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dx0, x0, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dx1, x1, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dx2, x2, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dz0, z0, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dz1, z1, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dz2, z2, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dz, z, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dnormx, normx, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dstrona, strona, ile * sizeof(char), cudaMemcpyHostToDevice);
	
	//	funkcja na GPU opisana w miejscu definicji
	crossection_cuda << <1, ile >> > (dy0, dy1, dy2, dx0, dx1, dx2, dz0, dz1, dz2, dstrona, dz, dnormx);

	//	i kopiujê z powrotem na CPU wektory danych po wykonaniu funkcji na GPU
	cudaMemcpy(y0, dy0, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y1, dy1, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y2, dy2, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(x0, dx0, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(x1, dx1, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(x2, dx2, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(z0, dz0, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(z1, dz1, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(z2, dz2, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(z, dz, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(normx, dnormx, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(strona, dstrona, ile * sizeof(char), cudaMemcpyDeviceToHost);

	//	zwalniam pamiêæ na GPU
	cudaFree(dy0);
	cudaFree(dy1);
	cudaFree(dy2);
	cudaFree(dx0);
	cudaFree(dx1);
	cudaFree(dx2);
	cudaFree(dz0);
	cudaFree(dz1);
	cudaFree(dz2);
	cudaFree(dz);
	cudaFree(dstrona);
	cudaFree(dnormx);

	//	ustawiam wartoœæ zmiennej z z powrotem na wartoœæ znalezionego minimum
	zorg = min;
	int index = 0;


	//	dla ka¿dej wyciêtej linii zapisujê j¹ pod odpowiednim indeksem w tablicy plane
	for (int i = 0; i < ile; i++)
	{
		//	obliczam index pod którym dana linia powinna zostaæ zapisana
		index = int((zorg - min) / accuracy);

		if (index < 0)
			index = 0;

		//	tworzê wskaŸnik na now¹ liniê
		line* abc = new line;

		//	i przypisujê do tej linii wartoœci obliczone na GPU
		abc[0].point[0].x = x0[i];
		abc[0].point[1].x = x1[i];
		abc[0].point[0].y = y0[i];
		abc[0].point[1].y = y1[i];
		abc[0].strona = strona[i];

		//	je¿eli strona ma wartoœæ inn¹ ni¿ 'T', to dodajê t¹ liniê do tablicy plane
		if (abc[0].strona != 'T')
			plane[index].plane.push_back(abc[0]);

		//	zwiêkszam zorg, aby nastêpna linia zosta³a zapisana pod innym indeksem,
		//	bo na podstawie zorg jest obliczany index
		zorg += accuracy;

		//	usuwam dynamicznie alokowan¹ zmienn¹ abc
		delete abc;

	}
}



//	konstruktor klasy Szukaj
Szukaj::Szukaj()
{
	//	obliczam ile warst bêdzie zawieraæ program po przeciêciu bry³y odpowiednimi p³aszczyznami 
	//	oddalonymi od siebie o wartoœæ zmiennej accuracy
	ile_warstw = (wczytywanie.max_z - wczytywanie.min_z) / wczytywanie.accuracy;
	plane_size = ile_warstw + 1;

	//	deklarujê dwie tablice struktur do przechowywania danych na podstawie obliczonego powy¿ej rozmiaru
	plane = new plane2D[plane_size];
	results = new point3D[plane_size];

}



//	konstruktor klasy Wczytywanie, który jest jednoczeœnie g³ównym dzia³aniem wykonywanym przez tê klasê
Wczytywanie::Wczytywanie()
{
	//	pobieramy od u¿ytkownika nazwê pliku oraz dok³adnoœæ wykonywanych obliczeñ
	string word, nazwa_pliku;

	cout << "Podaj nazwe pliku wraz z rozszerzeniem:     ";
	cin >> nazwa_pliku;

	cout << endl << "Podaj dokladnosc w milimetrach:     ";
	cin >> accuracy;
	cout << endl << endl;

	//	tworzê instancjê klasy fstream i otwieram plik o zadanej wy¿ej nazwie
	fstream solid;
	solid.open(nazwa_pliku, ios::in);

	float liczba;

	//	najpierw zliczam liczbê trójk¹tów zawartych w pliku, 
	//	¿eby wiedzieæ jak du¿¹ stworzyæ tablicê na dane
	while (solid.good() == true)
	{
		solid >> word;
		if (word == "normal")
			ile++;
	}

	//	tworzê tablicê struktór zawieraj¹c¹ pobrane z pliku dane
	triangle = new triangles[ile];

	//	zamykam i otwieram ponownie przeszukiwany plik
	solid.close();
	solid.open(nazwa_pliku, ios::in);


	//	poni¿sze instrukcje wynikaj¹ z budowy pliku formatu .stl, gdzie niektóre s³owa pomijam i 
	//	wyci¹gam tylko potrzebne do dzia³ania programu dane
	if (solid.good() == true)
	{
		solid >> word;
		solid >> word;
	}


	for (int i = 0; i < ile; i++)
	{
		for (int k = 0; k < 2; k++)
			solid >> word;

		solid >> liczba;
		triangle[i].normalna.x = liczba;
		solid >> liczba;
		triangle[i].normalna.y = liczba;
		solid >> liczba;
		triangle[i].normalna.z = liczba;

		for (int k = 0; k < 3; k++)
			solid >> word;

		solid >> liczba;
		triangle[i].first.x = liczba;
		solid >> liczba;
		triangle[i].first.y = liczba;
		solid >> liczba;
		triangle[i].first.z = liczba;

		solid >> word;
		solid >> liczba;
		triangle[i].second.x = liczba;
		solid >> liczba;
		triangle[i].second.y = liczba;
		solid >> liczba;
		triangle[i].second.z = liczba;

		solid >> word;
		solid >> liczba;
		triangle[i].third.x = liczba;
		solid >> liczba;
		triangle[i].third.y = liczba;
		solid >> liczba;
		triangle[i].third.z = liczba;

		for (int k = 0; k < 2; k++)
			solid >> word;

		if (i == 0)
		{
			min_z = max_z = triangle[i].first.z;
			min_y = max_y = triangle[i].first.y;
			min_x = max_x = triangle[i].first.x;
		}


		//	wyznaczanie najmniejszych i najwiêkszych wartoœci na poszczególnych osiach
		if (triangle[i].first.z < min_z)
			min_z = triangle[i].first.z;
		else if (triangle[i].second.z < min_z)
			min_z = triangle[i].second.z;
		else if (triangle[i].third.z < min_z)
			min_z = triangle[i].third.z;
		else if (triangle[i].first.z > max_z)
			max_z = triangle[i].first.z;
		else if (triangle[i].second.z > max_z)
			max_z = triangle[i].second.z;
		else if (triangle[i].third.z > max_z)
			max_z = triangle[i].third.z;

		if (triangle[i].first.y < min_y)
			min_y = triangle[i].first.y;
		else if (triangle[i].second.y < min_y)
			min_y = triangle[i].second.y;
		else if (triangle[i].third.y < min_y)
			min_y = triangle[i].third.y;
		else if (triangle[i].first.y > max_y)
			max_y = triangle[i].first.y;
		else if (triangle[i].second.y > max_y)
			max_y = triangle[i].second.y;
		else if (triangle[i].third.y > max_y)
			max_y = triangle[i].third.y;

		if (triangle[i].first.x < min_x)
			min_x = triangle[i].first.x;
		else if (triangle[i].second.x < min_x)
			min_x = triangle[i].second.x;
		else if (triangle[i].third.x < min_x)
			min_x = triangle[i].third.x;
		else if (triangle[i].first.x > max_x)
			max_x = triangle[i].first.x;
		else if (triangle[i].second.x > max_x)
			max_x = triangle[i].second.x;
		else if (triangle[i].third.x > max_x)
			max_x = triangle[i].third.x;

	}

	
	//	zaokr¹glam wartoœæ max_z do najbli¿szej mniejszej wielokrotnoœci zmiennej accuracy
	max_z = max_z / accuracy;
	max_z = floor(max_z);
	max_z = max_z * accuracy;

	//	zaokr¹glam wartoœæ min_z do najbli¿szej wiêkszej wielokrotnoœci zmiennej accuracy
	min_z = min_z / accuracy;
	min_z = ceil(min_z);
	min_z = min_z * accuracy;

	//	zamykam plik z danymi
	solid.close();

}



//	funkcja s³u¿¹ca do wypisania danych pobranych z pliku tekstowego formatu stl
void Wczytywanie::wypisz_dane()
{
	for (int i = 0; i < ile; i++)
	{
		cout << triangle[i].normalna.x << " " << triangle[i].normalna.y << " " << triangle[i].normalna.z << " "
			<< triangle[i].first.x << " " << triangle[i].first.y << " " << triangle[i].first.z << " "
			<< triangle[i].second.x << " " << triangle[i].second.y << " " << triangle[i].second.z << " "
			<< triangle[i].third.x << " " << triangle[i].third.y << " " << triangle[i].third.z << endl;
	}
}



