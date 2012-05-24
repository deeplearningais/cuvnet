#!/usr/bin/perl -w

sub start_client{
    my $cmd = shift;
    my $gpu = shift;
    return 0 == system(qq{tmux new-window 'while [ 1 -eq 1 ] ; do ./src/example/$cmd client $gpu ; sleep 10 ; done '});
}

sub git_pull{
    my $ret = system qq{ git pull };
    return $ret == 0;
}

sub make{
    my $client = shift;
    my $ret = system qq{ make -j $client };
    return $ret == 0;
}

sub get_n_gpus{
    return scalar split(/\n/, `ls /dev/nvidia?`);
}

sub run{
    my $client = shift;
    my $n_gpus = get_n_gpus();
    git_pull()     or die "could not git-pull!\n";
    make($client)  or die "could not make $client!\n";
    foreach (0 .. $n_gpus-1){
        print "starting on gpu $_\n";
        start_client($client, $_);
    }
}

run("stacked_auto_enc2");
