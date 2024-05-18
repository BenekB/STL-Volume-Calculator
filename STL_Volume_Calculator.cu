//	author: Benedykt Bela

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>



//	funkcja na GPU sumuj�ca ze sob� dwa wektory danych 
__global__ void add(float* R_side, float* L_side, int size,  float accuracy)
{
    int i = threadIdx.x;

	R_side[i] = R_side[i] - L_side[i];
	R_side[i] = R_side[i] * accuracy * accuracy;
}



//	funkcja na GPU s�u��ca do podzielenia odcinka na punkty zgodnie z zadan� dok�adno�ci�
//	dla kolejnych warto�ci zmiennej y punkt przeci�cia prostej y i odcinka mo�emy szuka� naraz
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


	//	tworz� nowy punkt przeci�cia odcinka z prost� y
	direct_point* punkt = new direct_point;
	point2D pomocniczy;		//	dodatkowy punkt usprawnia obliczenia


	int i = threadIdx.x;

	//	je�eli odcinek ko�czy si� i zaczyna w tym samym punkcie
	if (y0[i] == y1[i] && x0[i] == x1[i])
		punkt[0].strona = 'P';
	else
	{
		//	szukanie przeci�cia prostej y z zadanym odcinkiem
		pomocniczy.x = (x0[i] - x1[i]) * (y[i] - y1[i]) / (y1[i] - y0[i]);
		punkt[0].point.x = x1[i] + pomocniczy.x;
		punkt[0].strona = strona[i];
		punkt[0].point.y = y[i];
	}


	//	je�eli znaleziony punkt jest punktem skrajnym odcinka
	if (y0[i] < y1[i] && y1[i] == punkt[0].point.y)
		punkt[0].strona = 'P';
	if (y0[i] > y1[i] && y0[i] == punkt[0].point.y)
		punkt[0].strona = 'P';


	//	do wektora, kt�ry skopiuj� na CPU zapisuj� dane znalezionego punktu preci�cia
	y0[i] = punkt[0].point.y;
	x0[i] = punkt[0].point.x;
	strona[i] = punkt[0].strona;

}



//	funkcja na GPU s�u��ca do podzia�u tr�jk�ta na odcinki poprzec robienie przekroju 
//	przesuwaj�c� si� p�aszczyzn�
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


	//	tworz� struktury linia oraz punkt w przestrzeni tr�jwymiarowej, 
	//	kt�re b�d� pomocne przy dalszych obliczeniach
	line* linia = new line();
	point3D pomocniczyp;

	int j = 0;		//	potrzebne do zliczania kt�ry punkt aktualnie zapisuj� do struktury linia
	int i = threadIdx.x;

	
	//	poni�sze instrukcje warunkowe sprawdzaj� mi�dzy kt�rymi punktami znajduje si� obecnie 
	//	przeszukiwana p�aszczyzna, czyli kt�re odcinki tr�jk�ta b�dziemy przecina� oraz te odcinki
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


	//	sprawdzam w kt�r� stron� jest skierowana normalna danego tr�jk�ta, �eby wiedzie�
	//	gdzie jest �rodek badanego obiektu, a gdzie strona zewn�trzna
	if (normx[i] > 0)
		linia[0].strona = 'R';		//	R - z prawej
	else if (normx[i] < 0)
		linia[0].strona = 'L';		//	L - z lewej
	else
		linia[0].strona = 'T';		//	T oznacza tr�jk�t prostopad�y do osi y


	//	zapisuj� znalezione dane do wektor�w, kt�re skopiuj� na CPU
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


//	ka�dy tr�jk�t sk�ada si� z trzech punkt�w oraz wektora normalnego okre�laj�cego
//	kierunek wn�trza figury
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
	char strona;		//	strona oznacza kierunek wn�trza figury, kt�rej fragmentem
						//	jest dany odcinek
};


struct direct_point
{
	point2D point;
	char strona;
};


//	plane2D zawiera jedynie wektor sk�adaj�cy si� ze struktur linia
//	ta struktura umo�liwia zrobienie swego rodzaju tablicy wektor�w
struct plane2D
{
	vector<line> plane;
};


//	tablica wektor�w zawieraj�ca ostateczne punkty do obliczania obj�to�ci bry�y
struct final_points
{
	vector<direct_point> punkt;
};



//	klasa s�u��ca do wczytywania danych z pliku oraz przechowywania takich informacji
//	jak globalne minimum oraz maksimum w ka�dej p�aszczy�nie, ilo�� pobranych 
//	tr�jk�t�w, dok�adno�� oblcicze� oraz funkcj� do wypisywania posiadanych danych
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



//	g��wna klasa programu zawieraj�ca odpowiednie funkcje obliczaj�ce obj�to�� figury
class Szukaj {
public:
	Szukaj();

	//	wewn�trz g��wnej klasy programu inicjuj� instancj� klasy Wczytywanie, aby mie� 
	//	dost�p do jej danych 
	Wczytywanie wczytywanie;
	void cut_all_triangles();
	void cut_all_lines();
	void cut_triangle(triangles triangle);
	void cut_line(line linia, float g_min, float g_max);
	void wypisz_plane();
	float objetosc = 0;		//	zmienna zawieraj�ca warto�� obj�to�ci naszego obiektu


private:
	int ile_warstw;		//	liczba warstw przeci�ciu obiektu przesuwaj�c� si� p�aszczyzn� z
	float global_min = wczytywanie.min_z;		//	globalne minimum obiektu w osi z
	//	generalnie zmienne wykorzstywane w innych miejscach programu
	plane2D* plane;		
	point3D* results;
	float accuracy = wczytywanie.accuracy;
	int plane_size;
	final_points* final_point;

};



//	g��wna funkcja programu
int main()
{
	
	Szukaj szukaj;		//	inicjalizuj� instancj� klasy Szukaj
	szukaj.cut_all_triangles();		//	wykonuj� poci�cie wszystkich tr�jk�t�w
	szukaj.wypisz_plane();		//	wypisuje odcinki, kt�re sk�adaj� si� na przekroje obiektu
	szukaj.cut_all_lines();		//	przecina wszystkie dostepne linie i znajduje punkty przeci�cia

	cout << endl << szukaj.objetosc << endl << endl;		//	wypisuj� znalezion� obj�to�� figury

}



//	poni�sza funkcja przecina wszystkie linie tworz�ce przekroje r�wnoleg�e do p�aszczyzny z
//	i znajduje punkty przeci�cia na podstawie kt�rych oblicza obj�to�� bry�y
void Szukaj::cut_all_lines()
{
	float min, max;


	//	wykonuj� ci�cie i liczenie dla ka�dego przekroju r�wnoleg�ego do p�aszczyzny z
	for (int i = 0; i < plane_size; i++)
	{
		//	ustawiam pocz�tkow� warto�� dla zmiennych min i max
		min = plane[i].plane[0].point[0].y;
		max = min;

		//	dla ka�dego przekroju sprawdzam jego wsp�rz�dn� maksymaln� i minimaln� w osi y
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

		//	znalezione warto�ci min i max zaokr�glam do najbli�szej warto�ci y, kt�ra
		//	jest wielokrotno�ci� dok�adno�ci
		min = min / accuracy;
		min = ceil(min);
		min = min * accuracy;

		max = max / accuracy;
		max = floor(max);
		max = max * accuracy;

		//	tworz� now� tablic� punkt�w ostatecznych 
		final_point = new final_points[int((max - min) / accuracy + 1)];

		//	ka�d� lini� w danym przekroju przecinam i generuj� ostateczne punkty do obliczenia obj�to�ci
		for (int j = 0; j < plane[i].plane.size(); j++)
			cut_line(plane[i].plane[j], min, max);			

		int pomocnicza = int((max - min) / accuracy + 1);		//	liczba danych w tej iteracji p�tli
		
		//	wektory na CPU i GPU z warto�ciami do oblicze�
		float* R_side = new float[pomocnicza];
		float* d_R_side = new float[pomocnicza];
		float* L_side = new float[pomocnicza];
		float* d_L_side = new float[pomocnicza];

		//	wszystkie dane przegl�damy i zapisujemy te ze stron� 'R' do jedgeno wektora, 
		//	a te ze stron� 'L' do drugeigo wektora
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

		//	alokuj� miejsce na GPU
		cudaMalloc(& d_R_side, pomocnicza * sizeof(float));
		cudaMalloc(& d_L_side, pomocnicza * sizeof(float));

		//	oraz kopiuj� potrzebne dane
		cudaMemcpy(d_R_side, R_side, pomocnicza * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_L_side, L_side, pomocnicza * sizeof(float), cudaMemcpyHostToDevice);

		//	wykonuj� funkcj� dodawanie na GPU
		add << <1, pomocnicza >> > (d_R_side, d_L_side, pomocnicza, accuracy);

		//	i kopiuj� na CPU potrzebne dane
		cudaMemcpy(R_side, d_R_side, pomocnicza * sizeof(float), cudaMemcpyDeviceToHost);

		//	sumuj� skopiowane dane i dodaje do zmiennej obj�to��
		for (int j = 0; j < pomocnicza; j++)
			objetosc += R_side[j];

		//	usuwam dynamicznie alokowan� tablic� zmiennych 
		delete[] final_point;

	}
}



//	funkcja, kt�ra pobiera dane jednego obiektu line oraz globalne minimum i globalne maksimum,
//	a  zwraca punkty, kt�re s� punktami przeci�cia tej linii z p�aszczyznami prostopad�ymi do osi OY
//	ustawionymi w odleg�o�ci od siebie wynosz�cej warto�� zmiennej accuracy
void Szukaj::cut_line(line linia, float g_min, float g_max)
{
	float min, max;

	//	ustalam kt�ry koniec linii ma wi�ksz� wsp�rz�dn� y i przypisuj� odpowiednio do zmiennych min i max
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

	//	zaokr�glam warto�ci min i max w tej spos�b, �e nowe min jest najbli�sz�, ale wi�ksz� wielokrotno�ci� 
	//	zmiennej accuracy, natomiast max jest najbli�sz�, ale mniejsz� wielokrotno�ci� zmiennej accuracy
	min = min / accuracy;
	min = ceil(min);
	min = min * accuracy;

	max = max / accuracy;
	max = floor(max);
	max = max * accuracy;

	//	zmienne pomocnicze s�u��ce do oblicze�
	float yorg = min;
	float y_help = yorg;
	int index = 0;
	int ile = 0;

	//	zliczam ile wielokrotno�ci zmiennej accuracy mie�ci si� mi�dzy warto�ciami min i max,
	//	poniewa� tyle punkt�w funkcja musi wygenerowa�
	while (y_help <= max)
	{
		y_help += accuracy;
		ile++;
	}

	//	wektory zmiennych, do kt�rych zapisuj� odpowiednie warto�ci, kt�re p�niej skopiuj� na GPU i wykorzystam
	float* y0 = new float[ile];
	float* y1 = new float[ile];
	float* x0 = new float[ile];
	float* x1 = new float[ile];
	float* y = new float[ile];
	char* strona = new char[ile];

	//	odpowiedniki powy�szych wska�nik�w na GPU
	float* dy0 = new float[ile];
	float* dy1 = new float[ile];
	float* dx0 = new float[ile];
	float* dx1 = new float[ile];
	float* dy = new float[ile];
	char* dstrona = new char[ile];

	
	//	nadaj� zainicjowanym powy�ej wektorom odpowiednie warto�ci ze struktury linia
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

	//	alokuj� pami�� na GPU
	cudaMalloc(&dy0, ile * sizeof(float));
	cudaMalloc(&dy1, ile * sizeof(float));
	cudaMalloc(&dx0, ile * sizeof(float));
	cudaMalloc(&dx1, ile * sizeof(float));
	cudaMalloc(&dy, ile * sizeof(float));
	cudaMalloc(&dstrona, ile * sizeof(char));

	//	i kopiuj� na GPU dane z CPU
	cudaMemcpy(dy0, y0, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dy1, y1, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dx0, x0, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dx1, x1, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dy, y, ile * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dstrona, strona, ile * sizeof(char), cudaMemcpyHostToDevice);

	//	wywo�anie funkcji wykonywanej na GPU, kt�ra zosta�a opisana w miejscu definicji
	crossline_cuda << <1, ile >> > (dy0, dy1, dx0, dx1, dstrona, dy);

	//	kopiuj� dane z GPU z powrotem na CPU
	cudaMemcpy(y0, dy0, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y1, dy1, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(x0, dx0, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(x1, dx1, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, dy, ile * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(strona, dstrona, ile * sizeof(char), cudaMemcpyDeviceToHost);

	//	zwalniam pami�� na GPU
	cudaFree(dy0);
	cudaFree(dy1);
	cudaFree(dx0);
	cudaFree(dx1);
	cudaFree(dy);
	cudaFree(dstrona);

	//	ustawiam ponownie warto�� yorg na warto�� wcze�niej obliczonego min
	yorg = min;


	//	dla ka�dego wyci�tego punktu dodaj� go do tablicy punkt�w ostatecznych 
	for (int i = 0; i < ile; i++)
	{
		//	poni�sze kilka instrukcji ustala indeks w tablicy punkt�w ostatecznych do kt�rego dany punkt ma
		//	zosta� przypisany
		float help = yorg - g_min;

		if (help < 0)
			help = 1;

		index = int(help / accuracy);

		//	tworz� nowy wska�nik na obiekt direct_point
		direct_point* abc = new direct_point;

		//	przypisuj� do obiektu abc odpowiednie warto�ci z wektor�w danych obliczonych na GPU
		abc[0].point.x = x0[i];
		abc[0].point.y = y0[i];
		abc[0].strona = strona[i];

		//	je�eli strona danego punktu nie ma warto�ci 'P', to dodajemy ten punkt do wektora pod
		//	odpowiednim adresem tablicy final_point
		if (abc[0].strona != 'P')
			final_point[index].punkt.push_back(abc[0]);

		//	zwi�kszam aktualn� warto�� y, �eby nast�pny punkt zapisa� pod innym indeksem w 
		//	tablicy final_point, bo zmienna index jest ustalana w�a�nie na podstawie yorg
		yorg += accuracy;

		//	usuwam dynamicznie alokowan� zmienn� abc
		delete abc;
	}
}



//	funkcja nale�y do klasy Szukaj i s�u�y do wypisania w konsoli linii tworz�cych przekroje na poszczeg�lnych
//	p�aszczyznach prostopad�ych do osi OZ
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



//	ta kr�tka funkcja po prostu wywo�uje funkcj� cut_triangle dla wszystkich tr�jk�t�w wczytanych do programu
void Szukaj::cut_all_triangles()
{
	for (int i = 0; i < wczytywanie.ile; i++)
		cut_triangle(wczytywanie.triangle[i]);

}



//	poni�sza funkcja zwraca linie, kt�re powstaj� na skutek przeci�cia zadanego tr�jk�ta przez p�aszczyzn�
//	OXY przesuwan� z krokiem accuracy wzd�u� osi Z
void Szukaj::cut_triangle(triangles triangle)
{
	float min, max;

	//	ustalam minimaln� i maksymaln� wsp�rz�dn� tr�jk�ta w osi Z
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

	//	zaokr�glam zmienn� min do najbli�szej, wi�kszej od zmiennej min wielokrotno�ci zmiennej accuracy
	min = min / accuracy;
	min = ceil(min);
	min = min * accuracy;

	//	zaokr�glam zmienn� max do najbli�szej, mniejszej od zmiennej max wielokrotno�ci zmiennej accuracy
	max = max / accuracy;
	max = floor(max);
	max = max * accuracy;

	//	zmienne pomocnicze 
	int ile = 0;
	float zorg = min;
	float z_help = zorg;

	//	sprawdzam ile wielokrotno�ci zmiennej accuracy zmie�ci si� mi�dzy min i max, bo tyle linii 
	//	funkcja wytnie z tr�jk�ta
	while (z_help <= max)
	{
		z_help += accuracy;
		ile++;
	}

	//	wektory zmiennych do kt�rych zapisuj� odpowiednie dane, kt�re zostan� skopiowane na GPU
	//	i na ich podstawie GPU dokona oblicze�
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

	//	odpowiedniki powy�szych wska�nik�w wykorzystywane na GPU
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


	//	przypisuj� zadeklarowanym wektorom odpowiednie dane 
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

	//	alokuj� pami�� na GPU
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
	
	//	i kopiuj� potrzebne dane na GPU
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

	//	i kopiuj� z powrotem na CPU wektory danych po wykonaniu funkcji na GPU
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

	//	zwalniam pami�� na GPU
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

	//	ustawiam warto�� zmiennej z z powrotem na warto�� znalezionego minimum
	zorg = min;
	int index = 0;


	//	dla ka�dej wyci�tej linii zapisuj� j� pod odpowiednim indeksem w tablicy plane
	for (int i = 0; i < ile; i++)
	{
		//	obliczam index pod kt�rym dana linia powinna zosta� zapisana
		index = int((zorg - min) / accuracy);

		if (index < 0)
			index = 0;

		//	tworz� wska�nik na now� lini�
		line* abc = new line;

		//	i przypisuj� do tej linii warto�ci obliczone na GPU
		abc[0].point[0].x = x0[i];
		abc[0].point[1].x = x1[i];
		abc[0].point[0].y = y0[i];
		abc[0].point[1].y = y1[i];
		abc[0].strona = strona[i];

		//	je�eli strona ma warto�� inn� ni� 'T', to dodaj� t� lini� do tablicy plane
		if (abc[0].strona != 'T')
			plane[index].plane.push_back(abc[0]);

		//	zwi�kszam zorg, aby nast�pna linia zosta�a zapisana pod innym indeksem,
		//	bo na podstawie zorg jest obliczany index
		zorg += accuracy;

		//	usuwam dynamicznie alokowan� zmienn� abc
		delete abc;

	}
}



//	konstruktor klasy Szukaj
Szukaj::Szukaj()
{
	//	obliczam ile warst b�dzie zawiera� program po przeci�ciu bry�y odpowiednimi p�aszczyznami 
	//	oddalonymi od siebie o warto�� zmiennej accuracy
	ile_warstw = (wczytywanie.max_z - wczytywanie.min_z) / wczytywanie.accuracy;
	plane_size = ile_warstw + 1;

	//	deklaruj� dwie tablice struktur do przechowywania danych na podstawie obliczonego powy�ej rozmiaru
	plane = new plane2D[plane_size];
	results = new point3D[plane_size];

}



//	konstruktor klasy Wczytywanie, kt�ry jest jednocze�nie g��wnym dzia�aniem wykonywanym przez t� klas�
Wczytywanie::Wczytywanie()
{
	//	pobieramy od u�ytkownika nazw� pliku oraz dok�adno�� wykonywanych oblicze�
	string word, nazwa_pliku;

	cout << "Podaj nazwe pliku wraz z rozszerzeniem:     ";
	cin >> nazwa_pliku;

	cout << endl << "Podaj dokladnosc w milimetrach:     ";
	cin >> accuracy;
	cout << endl << endl;

	//	tworz� instancj� klasy fstream i otwieram plik o zadanej wy�ej nazwie
	fstream solid;
	solid.open(nazwa_pliku, ios::in);

	float liczba;

	//	najpierw zliczam liczb� tr�jk�t�w zawartych w pliku, 
	//	�eby wiedzie� jak du�� stworzy� tablic� na dane
	while (solid.good() == true)
	{
		solid >> word;
		if (word == "normal")
			ile++;
	}

	//	tworz� tablic� strukt�r zawieraj�c� pobrane z pliku dane
	triangle = new triangles[ile];

	//	zamykam i otwieram ponownie przeszukiwany plik
	solid.close();
	solid.open(nazwa_pliku, ios::in);


	//	poni�sze instrukcje wynikaj� z budowy pliku formatu .stl, gdzie niekt�re s�owa pomijam i 
	//	wyci�gam tylko potrzebne do dzia�ania programu dane
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


		//	wyznaczanie najmniejszych i najwi�kszych warto�ci na poszczeg�lnych osiach
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

	
	//	zaokr�glam warto�� max_z do najbli�szej mniejszej wielokrotno�ci zmiennej accuracy
	max_z = max_z / accuracy;
	max_z = floor(max_z);
	max_z = max_z * accuracy;

	//	zaokr�glam warto�� min_z do najbli�szej wi�kszej wielokrotno�ci zmiennej accuracy
	min_z = min_z / accuracy;
	min_z = ceil(min_z);
	min_z = min_z * accuracy;

	//	zamykam plik z danymi
	solid.close();

}



//	funkcja s�u��ca do wypisania danych pobranych z pliku tekstowego formatu stl
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



