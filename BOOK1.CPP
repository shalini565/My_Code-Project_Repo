
#include<fstream.h>
#include<conio.h>
#include<string.h>
#include<process.h>
#include<stdio.h>
#include<ctype.h>
class subject
{
public:
char s[8][60],sub[8][9][60];
subject()
{
strcpy(sub[0][0],"The Lost Symbol by DAN BROWN");
strcpy(sub[0][1],"Harry Potter Series by J.K.ROWLING");
strcpy(sub[0][2],"Endymion Spring by MATTHEW SKELTON");
strcpy(sub[0][3],"The Book Thief by MARKUS ZUSAK");
strcpy(sub[0][4],"Digital Fortress by DAN BROWN");
strcpy(sub[0][5],"Message in a bottle by NICHOLAS SPARKS");
strcpy(sub[0][6],"The Girl with the Dargon Tattoo by STIEG LARSSON");
strcpy(sub[0][7],"Blue Monday by NICCI FRENCH");
strcpy(sub[0][8],"If Tomorrow Comes by SIDNEY SHELDON");
strcpy(sub[0][9],"Morning,Noon and Night by SIDNEY SHELDON\t");
strcpy(sub[1][0],"Gulliver's Travels by JONATHAN SWIFT");
strcpy(sub[1][1],"Tales of Mystery And Imagination by POE");
strcpy(sub[1][2],"Mansfield Park by JANE AUSTEN");
strcpy(sub[1][3],"Treasure Island by ROBERT L STEVENSON ");
strcpy(sub[1][4],"Faust by JOHANN WOLFGANG VON GOETHE");
strcpy(sub[1][5],"Celtic Fairy Tales by JOSEPH JACOBS ");
strcpy(sub[1][6],"Frankenstein by MARY SHELLEY");
strcpy(sub[1][7],"Venus in Furs by LEOPOLD VON SACHER-MASOCH");
strcpy(sub[1][8],"Fairy Tales From The Arabian Nights by E. DIXON");
strcpy(sub[1][9],"Oliver Twist by CHARLES DICKENS\t");
strcpy(sub[2][0],"You Can Win");
strcpy(sub[2][1],"The Secret by RHONDA BRYNE");
strcpy(sub[2][2],"The Fountainhead by AYN RAND");
strcpy(sub[2][3],"The Grand Design by STEPHEN HAWKING");
strcpy(sub[2][4],"");
strcpy(sub[2][5],"");
strcpy(sub[2][6],"");
strcpy(sub[2][7],"");
strcpy(sub[2][8],"");
strcpy(sub[2][9],"A Hundred And One Days by ASNE SEIERSTAD\t");
strcpy(sub[3][0],"Contes de Noël by Vivanne de Mirmont");
strcpy(sub[3][1],"Regard Maternal by A Suzanne");
strcpy(sub[3][2],"Le Larron by A Henri");
strcpy(sub[3][3],"La Fuite by SAINT HENRY EXUPERY");
strcpy(sub[3][4],"Le Nourrisson de la Poupin by Louis");
strcpy(sub[3][5],"L'OBLIGEANTE PETITE FILLE");
strcpy(sub[3][6],"LA MOUCHE by Mélanie ");
strcpy(sub[3][7],"LA PETITE GOURMANDE by Marianne");
strcpy(sub[3][8],"I misteri del castello d'Udolfo by Ann Radcliffe");
strcpy(sub[3][9],"Jerry Spring by Jijé ");
strcpy(sub[4][0],"XIII by William Vance ");
strcpy(sub[4][1],"Adèle Blanc-Sec by Jacques Tardi");
strcpy(sub[4][2],"Blake and Mortimer by E.P. Jacobs");
strcpy(sub[4][3],"Blueberry by Jean-Michel Charlier and Jean Giraud");
strcpy(sub[4][4],"Boule and Bill by Jean Roba ");
strcpy(sub[4][5],"The Smurfs by Peyo ");
strcpy(sub[4][6],"The Adventures of Tintin by Hergé");
strcpy(sub[4][7],"Tanguy et Laverdure by Jean-Michel Charlier");
strcpy(sub[4][8],"Lucky Luke by Morris and René Goscinny");
strcpy(sub[4][9],"Marsupilami by André Franquin ");
strcpy(sub[5][0],"");
strcpy(sub[5][1],"");
strcpy(sub[5][2],"");
strcpy(sub[5][3],"");
strcpy(sub[5][4],"");
strcpy(sub[5][5],"");
strcpy(sub[5][6],"");
strcpy(sub[5][7],"");
strcpy(sub[5][8],"");
strcpy(sub[5][9],"\t");
strcpy(sub[6][0],"Interview with the Vampire by Anne Rice ");
strcpy(sub[6][1],"The House on Black Lake by Anastasia Blackwell");
strcpy(sub[6][2],"The Death of Halpin Frayser by Ambrose Bierce");
strcpy(sub[6][3],"The Vanishing American by Charles Beaumont");
strcpy(sub[6][4],"The Living Corpse by Vladimir Odoevsky");
strcpy(sub[6][5],"Wagner the Wehr-wolf by G.W.M. Reynolds");
strcpy(sub[6][6],"Black Bargain by Robert Bloch");
strcpy(sub[6][7],"Lost Souls by Poppy Z. Brite");
strcpy(sub[6][8],"Wuthering Heights by Emily Brontë ");
strcpy(sub[6][9],"Carmilla by J. Sheridan LeFanu\t");
strcpy(sub[7][0],"Oxford english textbook");
strcpy(sub[7][1],"Flamingo Textbook");
strcpy(sub[7][2],"Deitel and Deitel c++");
strcpy(sub[7][3],"");
strcpy(sub[7][4],"");
strcpy(sub[7][5],"");
strcpy(sub[7][6],"");
strcpy(sub[7][7],"");
strcpy(sub[7][8],"");
strcpy(sub[7][9],"\t");
strcpy(sub[8][0],"");
strcpy(sub[8][1],"");
strcpy(sub[8][2],"");
strcpy(sub[8][3],"");
strcpy(sub[8][4],"");
strcpy(sub[8][5],"");
strcpy(sub[8][6],"");
strcpy(sub[8][7],"");
strcpy(sub[8][8],"");
strcpy(sub[8][9],"La rousse french dictionairre\t");
strcpy(s[0],"FICTION\t");
strcpy(s[1],"CLASSICS\t");
strcpy(s[2],"NON FICTION\t");
strcpy(s[3],"OTHER LANGUAGE BOOKS\t");
strcpy(s[4],"COMICS\t");
strcpy(s[5],"AUTOGRAPHIES\t");
strcpy(s[6],"MYSTERY AND GOTHIC\t");
strcpy(s[7],"SCHOOL BOOKS\t");
strcpy(s[8],"DICTIONARY");

}
};
void show();
void show2();
class userlist
{
public:
int no;
char users[100][50];
};
class user
{
public:
int sbj[5][7],totp,totr;
char password[20];
void showpb()
{
cout<<"\n\n\n\n\n\n 'P' Purchase\n 'B' Back";
}
};
void uppercase(char *);
void pur_ret(char name[]);
void notification(user &);
void setdefault(user &);
void chkusrlst();
void chkbooklst();
void showuserlist();
int crtusr();
int showbooks();
int openacc();
void namecutter(char name[]);
int main()
{
textbackground(12);
textcolor(14);
chkusrlst();
chkbooklst();
mpage:
clrscr();
char ch;
cout<<"\n\n\t\t\tWELCOME TO CROSSBOOKS\n\n\n\n\n '1' See the ";
cout<<"lists of books\n '2' Create an account\n '3' Open an account";
cout<<"\n '4' See about no. of books purchased/returned\n";
cout<<" '5' See the list of accounts\n '6' Quit";show();
ch=getch();
switch(ch)
{
case '1': if(showbooks()==1)
{  goto mpage;
}
break;
case '2': crtusr();
goto mpage;
case '3': openacc();
goto mpage;
case '4': pur_ret("BOOKLIST");
goto mpage;
case '5': showuserlist();
goto mpage;
case '6': return 0;
default : goto mpage;
}
getch();
return 0;
}
int crtusr()
{
clrscr();
char name[50],reply;
ifstream fi;
user u;userlist list;
fi.open("USERLIST");fi.read((char *) &list,sizeof(list));fi.close();
if(list.no==99)
{ clrscr();cout<<"\n\tYou can't create any account.\n\t";
cout<<"Since the accoutn list is full,\n\tso you have to";
cout<<" delete an account";show2();getch();return 0;
}
cout<<"\n Enter your name(max 49 characters):\n ";
page2:
cin.getline(name,49);uppercase(name);
if(name[0]=='\0'||name[0]==' ')
{
clrscr();
cout<<"\n Please enter a valid name:";goto page2;
}
for(int i=0;name[i]!='\0';i++)
{
if(name[i]=='\\'||name[i]=='/'||name[i]==':'||name[i]=='*'||name[i]=='?'||name[i]=='\"'||name[i]=='<'||name[i]=='>'||name[i]=='|')
{ clrscr();
cout<<"\n Please enter a valid name:";goto page2;
}
}
fi.open(name);
if(fi.good())
{
clrscr();
cout<<"\n The name you have entered is already present!";
cout<<"\n Please enter another:\n ";goto page2;
}
fi.close();
page:  clrscr();
cout<<"\n Do you want to add a password? (y/n):";reply=getch();
if(reply=='Y'||reply=='y')
{ clrscr();
cout<<"\n Enter your password(max 19 characters):\n ";
cin.getline(u.password,19);
}
else if(reply=='N'||reply=='n')
{ strcpy(u.password,"########");
}
else{ goto page;}
setdefault(u);
ofstream fo(name);
fo.write((char *) &u,sizeof(u));fo.close();
strcpy(list.users[list.no],name);list.no++;
fo.open("USERLIST");fo.write((char *) &list,sizeof(list));fo.close();
clrscr();cout<<"\n\n\tCongratulations! ";
cout<<"You have created your account of name:\n\t'"<<name<<"'";
show2();getch();
return 0;
}
void setdefault(user &usr)
{
int i,j;
for(i=0;i<5;i++)
{
for(j=0;j<7;j++)
{
usr.sbj[i][j]=0;
}
}
usr.totp=0;usr.totr=0;
}
void chkusrlst()
{
ifstream usrlst;
usrlst.open("userlist");
if(!usrlst.good())
{
ofstream fo("userlist");
userlist list;
list.no=0;
fo.write((char *) &list,sizeof(list));
fo.close();
}
usrlst.close();
}
void chkbooklst()
{
ifstream booklst;int i,j;
booklst.open("booklist");
if(!booklst.good())
{
ofstream fo("booklist");
user bklst;
for(i=0;i<5;i++)
{
for(j=0;j<5;j++)
{
bklst.sbj[i][j]=1000;
}
}
for(i=0;i<5;i++)
{
bklst.sbj[i][5]=0;bklst.sbj[i][6]=0;
}
bklst.totp=0;bklst.totr=0;
fo.write((char *)&bklst,sizeof(bklst));
fo.close();
}
booklst.close();
}
int showbooks()
{
user guest;subject sb;char i,k,ch,ch1,ch2;int x,y,j;
page1:  clrscr();
ifstream fi("Booklist");
fi.read((char *) &guest,sizeof(guest));
fi.close();cout<<"\n\n\n\n\n\n\n";
for(x=0;x<5;x++)
{
cout<<" '"<<x+1<<"' "<<sb.s[x]<<"\n";
}
cout<<" 'B' Back";
show();ch=getch();
for(i='1',x=0;i<='5';i++,x++)
{
page2:  clrscr();
if(ch==i)
{
cout<<"\n\n\n\t\t"<<sb.s[x]<<"\n\n\n\n";
for(j=0;j<5;j++)
{
cout<<" \'"<<j+1<<"\' "<<sb.sub[x][j]<<"\t";
cout<<guest.sbj[x][j]<<" remaining\n";
}
cout<<" 'B' Back";show();ch1=getch();
for(k='1',y=0;k<='5';k++,y++)
{
if(ch1==k)
{
page3:  clrscr();
cout<<"\n\t\t\t\t"<<sb.sub[x][y];
guest.showpb();show();ch2=getch();
if(ch2=='P'||ch2=='p')
{
if(guest.sbj[x][y]>0)
{ guest.totp++;guest.sbj[x][5]++;guest.sbj[x][y]--;
notification(guest);
}
else
{
clrscr();
cout<<"\n\n\tYou can't purchase this book.\n\tMake ";
cout<<"sure that no. of copies of this book\n\tin ";
cout<<"this shop is greater than zero.";
show2();getch();goto page2;
}
}
else if(ch2=='B'||ch2=='b'){ goto page2;}
else{ goto page3;}
break;
}
}
if(ch1=='B'||ch1=='b'){ goto page1;}
else{ goto page2;}
}
}
if(ch=='B'||ch=='b'){ return 1;}
else{ goto page1;}
}
void notification(user &u)
{
clrscr();
ofstream f("BOOKLIST");
f.write((char *) &u,sizeof(u));
f.close();
cout<<"\n\n\n\n\n\n\n\tThank you for your purchasing this book. ";
cout<<"Next time you must create\n\tan account so that you can ";
cout<<"purchase more than one book and also can\n\treturn books";
cout<<" purchased from this shop.\n\n\n\n\n";
cout<<"\tNow press any key to exit";
getch();
exit(0);
}
void show()
{
cout<<"\n\n\n\n\n\tPress any of the keys given above!";
}
void show2()
{
cout<<"\n\n\n\n\n\tPress any key to back!";
}
void pur_ret(char name[])
{
clrscr();
user u;subject s;ifstream fi(name);int i;
fi.read((char *) &u,sizeof(u));fi.close();
cout<<"\n\n\n\n Subject\t\tPurchased\tReturned\n\n\n";
for(i=0;i<5;i++)
{
cout<<" "<<i+1<<"."<<" "<<s.s[i]<<"\t"<<u.sbj[i][5];
cout<<"\t\t"<<u.sbj[i][6]<<"\n";
}
cout<<" 6. "<<"Total\t\t"<<u.totp<<"\t\t"<<u.totr;show2();getch();
}
int openacc()
{
user u,bklst;userlist list;
char name[50],passw[20],reply;ofstream fo;
page1:clrscr();
cout<<"\n Enter your account name:\n ";cin.getline(name,49);
uppercase(name);ifstream fi(name);
if(!fi.good())
{ pagename:
clrscr();cout<<"\n\n Wrong account name!\n Enter again?(y/n)\n ";
reply=getch();
if(reply=='y'||reply=='Y'){ goto page1;}
else if(reply=='n'||reply=='N'){ return 0;}
else{ goto pagename;}
}
fi.read((char *) &u,sizeof(u));fi.close();
fi.open("USERLIST");fi.read((char *) &list,sizeof(list));fi.close();
int flag=0;
for(int s=0;s<list.no;s++)
{
if(!strcmp(name,list.users[s])){ flag=1;break;}
}
if(flag==0)
{ strcpy(list.users[list.no],name);list.no++;
fo.open("USERLIST");
fo.write((char *) &list,sizeof(list));fo.close();
}
if(!strcmp(u.password,"########")){ goto page3;}
page2: clrscr();cout<<"\n Enter your password:\n ";
cin.getline(passw,19);
if(strcmp(u.password,passw))
{ pagepass:
clrscr();cout<<"\n\n Wrong password!\n Enter again?(y/n)\n ";
reply=getch();
if(reply=='y'||reply=='Y'){ goto page2;}
else if(reply=='n'||reply=='N'){ return 0;}
else{ goto pagepass;}
}
page3: clrscr();fi.open("BOOKLIST");
fi.read((char *) &bklst,sizeof(bklst));fi.close();
cout<<"\n\n\n\tWelcome "<<name;
cout<<"\n\n\n\n '1' Purchase/return books\n '2' No. of books";
cout<<" you are having\n '3' No. of books purchased/retutned ";
cout<<"by you\n '4' Add/modify password\n '5' Remove password\n ";
cout<<"'6' Delete your account\n 'B' Back";show();reply=getch();
if(reply=='1')
{
subject sb;char i,k,ch,ch1,ch2;int x,y,j;
page31:  clrscr();
cout<<"\n\n\n\n\n\n\n";
for(x=0;x<5;x++)
{
cout<<" '"<<x+1<<"' "<<sb.s[x]<<"\n";
}
cout<<" 'B' Back";
show();ch=getch();
for(i='1',x=0;i<='5';i++,x++)
{
page32:  clrscr();
if(ch==i)
{
cout<<"\n\n\n\t\t"<<sb.s[x]<<"\n\n\n\n";
for(j=0;j<5;j++)
{
cout<<" \'"<<j+1<<"\' "<<sb.sub[x][j]<<"\t";
cout<<bklst.sbj[x][j]<<" remaining\n";
}
cout<<" 'B' Back";show();ch1=getch();
for(k='1',y=0;k<='5';k++,y++)
{
if(ch1==k)
{
page33:  clrscr();
cout<<"\n\t\t\t\t"<<sb.sub[x][y];
u.showpb();cout<<"\n 'R' Return";show();ch2=getch();
if(ch2=='P'||ch2=='p')
{
if(bklst.sbj[x][y]>0)
{
u.totp++;bklst.totp++;u.sbj[x][5]++;bklst.sbj[x][5]++;
u.sbj[x][y]++;bklst.sbj[x][y]--;fo.open("BOOKLIST");
fo.write((char *) &bklst,sizeof(bklst));fo.close();
fo.open(name);fo.write((char *) &u,sizeof(u));fo.close();
clrscr();cout<<"\n\n\tOK! You have purchased this book";
}
else
{
clrscr();
cout<<"\n\n\tYou can't purchase this book.\n\tMake ";
cout<<"sure that no. of copies of this book\n\tin ";
cout<<"this shop is greater than zero.";
}
show2();getch();
goto page32;
}
else if(ch2=='r'||ch2=='R')
{
if(u.sbj[x][y]>0)
{
u.totr++;bklst.totr++;u.sbj[x][6]++;bklst.sbj[x][6]++;
u.sbj[x][y]--;bklst.sbj[x][y]++;fo.open("BOOKLIST");
fo.write((char *) &bklst,sizeof(bklst));fo.close();
fo.open(name);fo.write((char *) &u,sizeof(u));fo.close();
clrscr();cout<<"\n\n\tOK! You have returned this book";
}
else
{
clrscr();
cout<<"\n\n\tYou can't return this book.\n\tMake ";
cout<<"sure that no. of copies of this book\n\tyou";
cout<<" are having is greater than zero.";
}
show2();getch();
goto page32;
}
else if(ch2=='B'||ch2=='b'){ goto page32;}
else{ goto page33;}
}
}
if(ch1=='B'||ch1=='b'){ goto page31;}
else{ goto page32;}
}
}
if(ch=='B'||ch=='b'){ goto page3;}
else{ goto page31;}
}
else if(reply=='2')
{
clrscr();
int i,j;
for(i=0;i<5;i++)
{ subject sb;
cout<<"\n "<<sb.s[i]<<"\n";
for(j=0;j<5;j++)
{
cout<<"  "<<j+1<<". "<<sb.sub[i][j]<<"\t"<<u.sbj[i][j];
cout<<" remaining\n";
}

}
show2();getch();goto page3;
}
else if(reply=='3')
{ pur_ret(name);
goto page3;
}
else if(reply=='4')
{ clrscr();cout<<"\n Enter new password(max 19 characters):\n ";
cin.getline(passw,19);strcpy(u.password,passw);clrscr();
cout<<"\n\tPassword modified succesfully!";show2();getch();
fo.open(name);fo.write((char *) &u,sizeof(u));fo.close();
goto page3;
}
else if(reply=='5')
{ page35:  clrscr();cout<<"\n Remove password?(y/n):";reply=getch();
if(reply=='Y'||reply=='y')
{ strcpy(u.password,"########");fo.open(name);
fo.write((char *) &u,sizeof(u));fo.close();clrscr();
cout<<"\n\tPassword removed successfully!";
show2();getch();goto page3;
}
else if(reply=='N'||reply=='n'){ goto page3;}
else{ goto page35;}
}
else if(reply=='6')
{ page36:  clrscr();
cout<<"\n Do you want to delete your account?(y/n):";reply=getch();
if(reply=='Y'||reply=='y')
{ clrscr();remove(name);
cout<<"\n\tYour account is deleted successfully!";show2();getch();
namecutter(name);
return 0;
}
else if(reply=='N'||reply=='n'){ goto page3;}
else{goto page36;}
}
else if(reply=='b'||reply=='B')
{ page3b:  clrscr();
cout<<"\n Do you want to exit from your account?(y/n):";
reply=getch();
if(reply=='Y'||reply=='y'){ return 0;}
else if(reply=='N'||reply=='n'){ goto page3;}
else{ goto page3b;}
}
else{ goto page3;}
}
void uppercase(char *a)
{
for(int i=0;a[i]!='\0';i++)
{
a[i]=toupper(a[i]);
}
}
void showuserlist()
{
clrscr();int i;userlist list;ifstream fi("USERLIST");
fi.read((char *) &list,sizeof(list));fi.close();
for(i=0;i<list.no;i++)
{
cout<<"\n "<<i+1<<". "<<list.users[i];
}
show2();getch();
}
void namecutter(char name[])
{
userlist u;char temp[50];int i;
ifstream fi("USERLIST");fi.read((char *) &u,sizeof(u));fi.close();
for(i=0;i<u.no-1;i++)
{
if(strcmp(u.users[i],name)==0)
{
strcpy(temp,u.users[i]);
strcpy(u.users[i],u.users[i+1]);
strcpy(u.users[i+1],temp);
}
}
u.no--;ofstream fo("USERLIST");
fo.write((char *) &u,sizeof(u));fo.close();
}
